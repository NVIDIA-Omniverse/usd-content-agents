# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline oracle for paired unrigged and rigged USD references.

The reference USD is evidence, never a production input.  OpenUSD imports stay
inside public functions so importing the joint-rigger contracts remains pxr-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse

from world_understanding.functions.physics.joint_rigger.mass_properties import (
    _canonicalize_quaternion,
    _lift_descendant_mass_frame,
)
from world_understanding.functions.physics.joint_rigger.models import (
    INPUT_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    ArticulationRootPlanV1,
    ArtifactIdentityV1,
    ColliderPlanV1,
    FieldProvenanceV1,
    JointAnchorV1,
    JointDriveV1,
    JointFrictionV1,
    JointLimitV1,
    JointPlanV1,
    JointRiggerContractError,
    JointRiggerInputV1,
    JointRiggerPlanV1,
    JointTopologyV1,
    MassPropertiesV1,
    RigidBodyPlanV1,
    canonical_json,
)
from world_understanding.functions.physics.joint_rigger.opaque_dependencies import (
    OPAQUE_DEPENDENCY_EXTENSIONS,
    OpaqueDependencyError,
    opaque_local_references,
    resolve_local_reference,
)

_SUPPORTED_JOINT_TYPES = frozenset({"revolute", "prismatic", "spherical"})
_AXIS_FRAME_BASES = {
    "x": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "y": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "z": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}
_FRAME_TOLERANCE = 1e-5
# Joint frame rotations are stored as Quatf values.  This admits ordinary
# float32 round-trip error while still rejecting meaningfully scaled inputs.
_JOINT_FRAME_QUATERNION_NORM_TOLERANCE = 1e-5
_JOINT_FRAME_QUATERNION_ZERO_NORM_TOLERANCE = 1e-12
_STAGE_METADATA_TOLERANCE = 1e-12
_BUNDLE_SCHEMA = "world-understanding-usd-dependency-bundle-v3"
_MAX_DEPENDENCY_VALIDATION_PASSES = 256
_MAX_PHYSICS_MATERIAL_COLLECTION_DEPTH = 64
_MAX_PHYSICS_MATERIAL_COLLECTION_DEFINITIONS = 256
_AR_ASSET_READ_CHUNK_SIZE = 1024 * 1024
_MAX_OPAQUE_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_OPAQUE_DEPENDENCY_FILES = 256
_MAX_OPAQUE_DEPENDENCY_REFERENCES = 4096
_PRECOMPOSITION_LAYER_FORMATS = frozenset({"usd", "usda", "usdc"})
_PRECOMPOSITION_EXPORT_FORMATS = frozenset({"usda", "usdc"})
_PRECOMPOSITION_LAYER_SUFFIXES = frozenset(
    f".{format_name}" for format_name in _PRECOMPOSITION_LAYER_FORMATS
)
_PrecompositionFileState = tuple[int, int, int, int, int, int, int]
_PrecompositionSymlinkHop = tuple[Path, _PrecompositionFileState, str]
_UNREPRESENTED_JOINT_PROPERTY_PREFIXES = (
    "state:",
    "physxMimicJoint:",
)
_PHYSX_JOINT_SCHEMA = "PhysxJointAPI"
_PHYSX_MAX_JOINT_VELOCITY = "physxJoint:maxJointVelocity"
_PHYSX_JOINT_FRICTION = "physxJoint:jointFriction"
_BASE_JOINT_FALLBACK_PROPERTIES = {
    "physics:jointEnabled": True,
    "physics:collisionEnabled": False,
    "physics:excludeFromArticulation": False,
}
_UNREPRESENTED_JOINT_BREAK_PROPERTIES = (
    "physics:breakForce",
    "physics:breakTorque",
)
_SPHERICAL_LIMIT_PROPERTIES = (
    "physics:coneAngle0Limit",
    "physics:coneAngle1Limit",
)
_MASS_API_PROPERTIES = (
    "physics:mass",
    "physics:density",
    "physics:centerOfMass",
    "physics:diagonalInertia",
    "physics:principalAxes",
)
_UNREPRESENTED_MASS_PROPERTIES = ("physics:density",)
_RIGID_BODY_FALLBACK_PROPERTIES: dict[
    str,
    bool | tuple[float, float, float],
] = {
    "physics:rigidBodyEnabled": True,
    "physics:kinematicEnabled": False,
    "physics:startsAsleep": False,
    "physics:velocity": (0.0, 0.0, 0.0),
    "physics:angularVelocity": (0.0, 0.0, 0.0),
}
_COLLISION_FALLBACK_PROPERTIES: dict[str, bool | tuple[float, float, float]] = {
    "physics:collisionEnabled": True,
}
_COLLISION_API_PROPERTIES = frozenset(
    {
        "physics:collisionEnabled",
        "physics:approximation",
    }
)
_COLLIDER_NON_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "purpose",
        "visibility",
        "xformOpOrder",
    }
)
_ColliderApproximation = Literal[
    "none",
    "convexHull",
    "convexDecomposition",
    "sdf",
]
_SUPPORTED_COLLIDER_APPROXIMATIONS = frozenset(
    {"none", "convexHull", "convexDecomposition", "sdf"}
)
_RIGGER_OWNED_PHYSICS_PROPERTIES = frozenset(
    {
        *_RIGID_BODY_FALLBACK_PROPERTIES,
        *_MASS_API_PROPERTIES,
        *_COLLISION_API_PROPERTIES,
    }
)
_DRIVE_PROPERTY_PREFIX = "drive:"
_DRIVE_PROPERTY_SUFFIXES = frozenset(
    {
        "type",
        "stiffness",
        "damping",
        "maxForce",
        "targetPosition",
        "targetVelocity",
    }
)
_DIRECT_PHYSICS_MATERIAL_BINDING = "material:binding:physics"
_COLLECTION_PHYSICS_MATERIAL_BINDING_PREFIX = "material:binding:collection:physics:"
_REPRESENTED_JOINT_PROPERTIES = frozenset(
    {
        "physics:axis",
        "physics:body0",
        "physics:body1",
        "physics:breakForce",
        "physics:breakTorque",
        "physics:collisionEnabled",
        "physics:coneAngle0Limit",
        "physics:coneAngle1Limit",
        "physics:excludeFromArticulation",
        "physics:jointEnabled",
        "physics:localPos0",
        "physics:localPos1",
        "physics:localRot0",
        "physics:localRot1",
        "physics:lowerLimit",
        "physics:upperLimit",
        _PHYSX_MAX_JOINT_VELOCITY,
        _PHYSX_JOINT_FRICTION,
    }
)


@dataclass(frozen=True)
class _ResolvedUsdDependency:
    """One dependency resolved by OpenUSD for identity and path protection."""

    kind: Literal["layer", "asset", "opaque_asset"]
    identifier: str
    lexical_path: Path | None
    local_path: Path | None
    package_relative: bool
    read_identifier: str | None = None
    captured_sha256: str | None = None


@dataclass(frozen=True)
class _ProjectedLocalFile:
    """One descriptor-copied local file retained by a composition projection."""

    lexical_path: Path
    backing_path: Path
    projected_path: Path
    expected_state: _PrecompositionFileState
    symlink_hops: tuple[_PrecompositionSymlinkHop, ...]
    sha256: str


@dataclass(frozen=True)
class _ProjectedLocator:
    """One clearly local authored locator discovered without composition."""

    path: Path | None
    format_hint: str | None
    inspect_layer: bool
    rewritten: str


@dataclass
class _UsdCompositionProjection:
    """Private mirrored closure used as the only OpenUSD composition source."""

    mirror_root: Path
    files: dict[Path, _ProjectedLocalFile]
    closures: dict[Path, set[Path]]
    layer_dependencies: dict[tuple[Path, str | None], tuple[_ProjectedLocator, ...]]
    opaque_dependencies: dict[Path, set[Path]] = dataclass_field(default_factory=dict)

    def projected_path(self, path: Path) -> Path:
        absolute = Path(os.path.abspath(path.expanduser()))
        return self.mirror_root.joinpath(*absolute.parts[1:])

    def original_identifier(self, identifier: str, *, Ar: Any, Sdf: Any) -> str:
        """Map a projected resolved identifier back to its authored filesystem."""

        try:
            authored_path, arguments = Sdf.Layer.SplitIdentifier(identifier)
        except Exception:
            return identifier
        package_inner: str | None = None
        outer = authored_path
        if Ar.IsPackageRelativePath(outer):
            outer, package_inner = Ar.SplitPackageRelativePathOuter(outer)
        if not outer or "://" in outer:
            return identifier
        projected = Path(outer).expanduser()
        if not projected.is_absolute():
            return identifier
        try:
            relative = projected.relative_to(self.mirror_root)
        except ValueError:
            return identifier
        original_path = Path(projected.anchor).joinpath(*relative.parts)
        rebuilt = str(original_path)
        if package_inner is not None:
            rebuilt = Ar.JoinPackageRelativePath(rebuilt, package_inner)
        if arguments:
            rebuilt = Sdf.Layer.CreateIdentifier(rebuilt, arguments)
        return rebuilt

    def require_unchanged(
        self,
        root: Path,
        *,
        code: str,
        root_code: str | None = None,
    ) -> None:
        """Require every local file in one projected closure to stay identical."""

        absolute_root = Path(os.path.abspath(root.expanduser()))
        for lexical_path in sorted(
            self.closures.get(absolute_root, set()),
            key=lambda item: item.as_posix(),
        ):
            record = self.files[lexical_path]
            observed_code = (
                root_code
                if root_code is not None and lexical_path == absolute_root
                else code
            )
            try:
                descriptor, state, backing_path, symlink_hops = (
                    _open_precomposition_regular_file(lexical_path)
                )
                try:
                    digest = _precomposition_descriptor_sha256(
                        descriptor,
                        size=state[4],
                        label=str(lexical_path),
                    )
                finally:
                    os.close(descriptor)
            except JointRiggerContractError as exc:
                raise JointRiggerContractError(
                    observed_code,
                    f"USD dependency changed while reading {lexical_path}: {exc}",
                ) from exc
            if (
                state != record.expected_state
                or backing_path != record.backing_path
                or symlink_hops != record.symlink_hops
                or digest != record.sha256
            ):
                _fail(
                    observed_code,
                    f"USD dependency changed while reading {lexical_path}",
                )


@dataclass(frozen=True)
class _CapturedDependencyStructure:
    """One canonical dependency locator captured without reading its bytes."""

    kind: str
    locator: str
    backing_path: Path | None
    package_inner_locator: str | None


@dataclass(frozen=True)
class _CapturedDependencyIdentityRecord:
    """One canonical dependency identity finalized from retained descriptors."""

    kind: str
    locator: str
    sha256: str
    backing_path: Path | None

    def identity_record(self) -> dict[str, str]:
        """Return only fields serialized into the public dependency identity."""

        return {
            "kind": self.kind,
            "locator": self.locator,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _PhysicsMaterialBindingTarget:
    """One replay-relevant material binding target."""

    material_path: str
    collection_prim_path: str | None = None
    collection_instance: str | None = None


_DescendantMassReplayStatus = Literal[
    "matching_preexisting",
    "reference_only",
    "source_conflict",
]


@dataclass(frozen=True)
class _DescendantMassEvidence:
    """One nearest-owner descendant mass record and its replay disposition."""

    prim: Any
    reference_evidence: tuple[str, ...]
    source_evidence: tuple[str, ...]
    replay_status: _DescendantMassReplayStatus


def identify_usd_artifact(
    path: str | Path,
    *,
    uri: str,
) -> ArtifactIdentityV1:
    """Return a stable root-and-dependency identity for one composed USD.

    The dependency digest covers the stage's complete used-layer/asset closure
    plus reachable local MDL and MaterialX descendants. The root and dependency
    closure are hashed again before this function returns, so a file that
    changes while identity is being established fails closed.
    """

    from pxr import Usd

    artifact_path = Path(path)
    root_sha256 = _file_sha256(artifact_path, code="artifact_missing")
    with _usd_composition_projection((artifact_path,)) as projection:
        projected_path = projection.projected_path(artifact_path)
        _require_projected_root_matches_hash(
            artifact_path,
            projection=projection,
            expected_sha256=root_sha256,
            code="artifact_mutated",
        )
        stage = _open_stage(
            projected_path,
            Usd=Usd,
            label="artifact",
            display_path=artifact_path,
        )
        dependencies = _enumerate_usd_dependencies(
            artifact_path,
            projection=projection,
            root_mutated_code="artifact_mutated",
            dependency_mutated_code="artifact_dependency_mutated",
        )
        projection.require_unchanged(
            artifact_path,
            code="artifact_dependency_mutated",
            root_code="artifact_mutated",
        )
        identity = _artifact_identity(
            artifact_path,
            uri,
            stage,
            root_sha256=root_sha256,
            dependency_records=_dependency_identity_records(
                artifact_path,
                dependencies,
            ),
        )
        _require_artifact_identity_unchanged(
            artifact_path,
            stage,
            identity,
            dependencies=dependencies,
            projection=projection,
            missing_code="artifact_missing",
            root_mutated_code="artifact_mutated",
            dependency_mutated_code="artifact_dependency_mutated",
        )
        return identity


@contextmanager
def _usd_composition_projection(
    paths: Sequence[Path],
) -> Iterator[_UsdCompositionProjection]:
    """Retain a private regular-file mirror for all clearly local dependencies."""

    with tempfile.TemporaryDirectory(prefix="joint-rigger-composition-") as directory:
        mirror_root = Path(directory) / "absolute"
        mirror_root.mkdir(mode=0o700)
        projection = _UsdCompositionProjection(
            mirror_root=mirror_root,
            files={},
            closures={},
            layer_dependencies={},
        )
        for path in paths:
            _populate_projection_root(projection, path)
        for record in projection.files.values():
            record.projected_path.chmod(0o400)
        yield projection


def _preflight_local_dependency_locators(path: Path) -> None:
    """Build and discard one complete private projection as a validation pass."""

    with _usd_composition_projection((path,)):
        return


def _populate_projection_root(
    projection: _UsdCompositionProjection,
    path: Path,
) -> None:
    """Copy one root closure and rewrite absolute local locators in the mirror."""

    from pxr import Ar, Sdf, Usd, UsdUtils  # noqa: F401

    root = Path(os.path.abspath(path.expanduser()))
    closure = projection.closures.setdefault(root, set())
    pending: list[tuple[Path, str | None, bool]] = [(root, None, True)]
    expanded_layers: set[tuple[Path, str | None]] = set()
    inspected_layers = 0
    while pending:
        layer_path, format_hint, is_root = pending.pop()
        layer_path = Path(os.path.abspath(layer_path.expanduser()))
        record = _copy_precomposition_file(projection, layer_path)
        closure.add(layer_path)
        suffix = _precomposition_layer_suffix(layer_path, format_hint=format_hint)
        if suffix is None:
            if is_root and layer_path.suffix.lower() == ".usdz":
                _require_cached_package_matches_projection(
                    layer_path,
                    record,
                    Sdf=Sdf,
                )
            continue
        key = (layer_path, format_hint)
        if key in expanded_layers:
            continue
        expanded_layers.add(key)
        dependencies = projection.layer_dependencies.get(key)
        if dependencies is None:
            inspected_layers += 1
            if inspected_layers > _MAX_DEPENDENCY_VALIDATION_PASSES:
                _fail(
                    "artifact_dependency_preflight_failed",
                    "authored local USD dependency validation did not converge "
                    f"after {_MAX_DEPENDENCY_VALIDATION_PASSES} layers for {path}",
                )
            dependencies = _inspect_precomposition_layer(
                projection,
                record,
                owner_path=layer_path,
                format_hint=format_hint,
                allow_invalid_layer=is_root,
                Ar=Ar,
                Sdf=Sdf,
                UsdUtils=UsdUtils,
            )
            projection.layer_dependencies[key] = dependencies
        for dependency in reversed(dependencies):
            if dependency.path is None:
                continue
            dependency_path = Path(os.path.abspath(dependency.path.expanduser()))
            _copy_precomposition_file(projection, dependency_path)
            closure.add(dependency_path)
            if dependency.inspect_layer:
                pending.append((dependency_path, dependency.format_hint, False))
    _populate_opaque_projection_closure(
        projection,
        root=root,
        closure=closure,
    )


def _populate_opaque_projection_closure(
    projection: _UsdCompositionProjection,
    *,
    root: Path,
    closure: set[Path],
) -> None:
    """Descriptor-copy every reachable local MDL/MaterialX dependency."""

    pending = sorted(
        (
            (path, _opaque_projection_allowed_root(projection, path))
            for path in closure
            if path.suffix.lower() in OPAQUE_DEPENDENCY_EXTENSIONS
        ),
        key=lambda item: item[0].as_posix(),
    )
    visited: set[Path] = set()
    discovered: set[Path] = set()
    reference_count = 0
    while pending:
        document, allowed_root = pending.pop(0)
        if document in visited:
            continue
        if len(visited) >= _MAX_OPAQUE_DEPENDENCY_FILES:
            _fail(
                "artifact_dependency_preflight_failed",
                "opaque material dependency closure exceeds the "
                f"{_MAX_OPAQUE_DEPENDENCY_FILES}-file limit for {root}",
            )
        record = _copy_precomposition_file(
            projection,
            document,
            opaque_allowed_root=allowed_root,
        )
        try:
            text = record.projected_path.read_bytes().decode("utf-8")
            references = opaque_local_references(text, document=document)
        except UnicodeDecodeError as exc:
            raise JointRiggerContractError(
                "artifact_dependency_preflight_failed",
                f"opaque material dependency is not UTF-8 text: {document}",
            ) from exc
        except OpaqueDependencyError as exc:
            raise JointRiggerContractError(
                "artifact_dependency_preflight_failed",
                str(exc),
            ) from exc
        visited.add(document)
        reference_count += len(references)
        if reference_count > _MAX_OPAQUE_DEPENDENCY_REFERENCES:
            _fail(
                "artifact_dependency_preflight_failed",
                "opaque material dependency closure exceeds the "
                f"{_MAX_OPAQUE_DEPENDENCY_REFERENCES}-reference limit for {root}",
            )
        for value in references:
            try:
                target = resolve_local_reference(
                    document,
                    value,
                    allowed_root=allowed_root,
                )
            except OpaqueDependencyError as exc:
                raise JointRiggerContractError(
                    "artifact_dependency_preflight_failed",
                    str(exc),
                ) from exc
            _copy_precomposition_file(
                projection,
                target,
                opaque_allowed_root=allowed_root,
            )
            closure.add(target)
            discovered.add(target)
            if (
                target.suffix.lower() in OPAQUE_DEPENDENCY_EXTENSIONS
                and target not in visited
                and all(target != item[0] for item in pending)
            ):
                pending.append((target, allowed_root))
        pending.sort(key=lambda item: item[0].as_posix())
    projection.opaque_dependencies[root] = discovered


def _opaque_projection_allowed_root(
    projection: _UsdCompositionProjection,
    path: Path,
) -> Path:
    """Anchor one opaque document to its containing artifact tree."""

    lexical_path = Path(os.path.abspath(path.expanduser()))
    candidates: list[Path] = []
    for projection_root in projection.closures:
        bundle_root = Path(os.path.abspath(projection_root.expanduser())).parent
        try:
            lexical_path.relative_to(bundle_root)
        except ValueError:
            continue
        candidates.append(bundle_root)
    if candidates:
        return max(candidates, key=lambda item: len(item.parts))
    # A resolver may explicitly bind an external opaque document. Treat that
    # document's lexical directory as its separate, narrowly anchored tree.
    return lexical_path.parent


def _require_opaque_projection_file_constraints(
    *,
    path: Path,
    expected_state: _PrecompositionFileState,
    symlink_hops: Sequence[_PrecompositionSymlinkHop],
    allowed_root: Path | None,
) -> None:
    """Bound opaque documents and reject symlink traversal in their tree.

    General USD dependency identity intentionally records stable symlink aliases.
    Opaque documents and parsed descendants are a narrower trust boundary. This
    check consumes metadata from the retained source descriptor before any bytes
    are copied or hashed, avoiding a separate check-then-open race.
    """

    lexical_path = Path(os.path.abspath(path.expanduser()))
    if (
        lexical_path.suffix.lower() in OPAQUE_DEPENDENCY_EXTENSIONS
        and expected_state[4] > _MAX_OPAQUE_DOCUMENT_BYTES
    ):
        _fail(
            "artifact_dependency_preflight_failed",
            "opaque material dependency exceeds the "
            f"{_MAX_OPAQUE_DOCUMENT_BYTES}-byte limit: {lexical_path}",
        )
    if allowed_root is None:
        return
    normalized_root = Path(os.path.abspath(allowed_root.expanduser()))
    try:
        lexical_path.relative_to(normalized_root)
    except ValueError:
        _fail(
            "artifact_dependency_preflight_failed",
            "opaque material dependency escapes its artifact tree: "
            f"{lexical_path} (root {normalized_root})",
        )
    for hop_path, _, _ in symlink_hops:
        try:
            hop_path.relative_to(normalized_root)
        except ValueError:
            continue
        _fail(
            "artifact_dependency_preflight_failed",
            "opaque material dependency traverses a symlink in its artifact "
            f"tree: {lexical_path} via {hop_path}",
        )


def _copy_precomposition_file(
    projection: _UsdCompositionProjection,
    path: Path,
    *,
    opaque_allowed_root: Path | None = None,
) -> _ProjectedLocalFile:
    """Copy exact bytes from a validated descriptor into the private mirror."""

    lexical_path = Path(os.path.abspath(path.expanduser()))
    if (
        opaque_allowed_root is None
        and lexical_path.suffix.lower() in OPAQUE_DEPENDENCY_EXTENSIONS
    ):
        opaque_allowed_root = _opaque_projection_allowed_root(
            projection,
            lexical_path,
        )
    existing = projection.files.get(lexical_path)
    if existing is not None:
        _require_opaque_projection_file_constraints(
            path=lexical_path,
            expected_state=existing.expected_state,
            symlink_hops=existing.symlink_hops,
            allowed_root=opaque_allowed_root,
        )
        return existing
    descriptor, expected_state, backing_path, symlink_hops = (
        _open_precomposition_regular_file(lexical_path)
    )
    projected_path = projection.projected_path(lexical_path)
    target_descriptor = -1
    created = False
    try:
        _require_opaque_projection_file_constraints(
            path=lexical_path,
            expected_state=expected_state,
            symlink_hops=symlink_hops,
            allowed_root=opaque_allowed_root,
        )
        projected_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        target_descriptor = os.open(projected_path, flags, 0o600)
        created = True
        digest = hashlib.sha256()
        offset = 0
        while offset < expected_state[4]:
            chunk = os.pread(
                descriptor,
                min(_AR_ASSET_READ_CHUNK_SIZE, expected_state[4] - offset),
                offset,
            )
            if not chunk:
                _fail(
                    "dependency_artifact_invalid",
                    f"local USD dependency changed while projected: {lexical_path}",
                )
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(target_descriptor, remaining)
                if written <= 0:  # pragma: no cover - regular-file OS invariant
                    raise OSError(
                        f"short write while projecting local dependency {lexical_path}"
                    )
                remaining = remaining[written:]
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            _fail(
                "dependency_artifact_invalid",
                f"local USD dependency grew while projected: {lexical_path}",
            )
        os.fsync(target_descriptor)
        _require_precomposition_file_unchanged(
            backing_path,
            descriptor=descriptor,
            expected_state=expected_state,
        )
        _require_precomposition_symlinks_unchanged(symlink_hops)
        record = _ProjectedLocalFile(
            lexical_path=lexical_path,
            backing_path=backing_path,
            projected_path=projected_path,
            expected_state=expected_state,
            symlink_hops=symlink_hops,
            sha256=digest.hexdigest(),
        )
        projection.files[lexical_path] = record
        return record
    except BaseException:
        if created:
            projected_path.unlink(missing_ok=True)
        raise
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        os.close(descriptor)


def _inspect_precomposition_layer(
    projection: _UsdCompositionProjection,
    record: _ProjectedLocalFile,
    *,
    owner_path: Path,
    format_hint: str | None,
    allow_invalid_layer: bool,
    Ar: Any,
    Sdf: Any,
    UsdUtils: Any,
) -> tuple[_ProjectedLocator, ...]:
    """Parse one projected layer without allowing OpenUSD to touch live paths."""

    projected_identifier = str(record.projected_path)
    if format_hint:
        projected_identifier = Sdf.Layer.CreateIdentifier(
            projected_identifier,
            {"format": format_hint},
        )
    try:
        layer = Sdf.Layer.OpenAsAnonymous(projected_identifier)
    except Exception as exc:
        if allow_invalid_layer:
            return ()
        raise JointRiggerContractError(
            "artifact_dependency_preflight_failed",
            f"could not inspect local USD dependency layer {owner_path}: {exc}",
        ) from exc
    if layer is None:
        if allow_invalid_layer:
            return ()
        _fail(
            "artifact_dependency_preflight_failed",
            f"could not inspect local USD dependency layer: {owner_path}",
        )
    _require_cached_layer_matches_projection(
        owner_path,
        layer,
        format_hint=format_hint,
        Sdf=Sdf,
    )
    dependencies: dict[tuple[str, str | None, bool], _ProjectedLocator] = {}
    rewritten = False

    def project_locator(asset_path: str) -> str:
        nonlocal rewritten
        if not asset_path:
            return asset_path
        dependency = _projected_local_locator(
            str(asset_path),
            owner_path=owner_path,
            projection=projection,
            Ar=Ar,
            Sdf=Sdf,
        )
        if dependency is None:
            return asset_path
        key = (
            str(dependency.path or ""),
            dependency.format_hint,
            dependency.inspect_layer,
        )
        dependencies[key] = dependency
        rewritten = rewritten or dependency.rewritten != asset_path
        return dependency.rewritten

    try:
        UsdUtils.ModifyAssetPaths(
            layer,
            project_locator,
            keepEmptyPathsInArrays=True,
        )
    except Exception as exc:
        raise JointRiggerContractError(
            "artifact_dependency_preflight_failed",
            f"could not enumerate authored paths in {owner_path}: {exc}",
        ) from exc
    if rewritten:
        try:
            file_format = str(layer.GetFileFormat().formatId)
            arguments = (
                {"format": file_format}
                if file_format in _PRECOMPOSITION_EXPORT_FORMATS
                else {}
            )
            exported = layer.Export(str(record.projected_path), args=arguments)
        except Exception as exc:
            raise JointRiggerContractError(
                "artifact_dependency_preflight_failed",
                f"could not write projected USD layer {owner_path}: {exc}",
            ) from exc
        if not exported:
            _fail(
                "artifact_dependency_preflight_failed",
                f"could not write projected USD layer: {owner_path}",
            )
    return tuple(
        dependencies[key]
        for key in sorted(
            dependencies, key=lambda item: (item[0], item[1] or "", item[2])
        )
    )


def _projected_local_locator(
    locator: str,
    *,
    owner_path: Path,
    projection: _UsdCompositionProjection,
    Ar: Any,
    Sdf: Any,
) -> _ProjectedLocator | None:
    """Resolve one clearly local locator and map it into the private mirror."""

    try:
        authored_path, arguments = Sdf.Layer.SplitIdentifier(locator)
    except Exception as exc:
        raise JointRiggerContractError(
            "artifact_dependency_preflight_failed",
            f"could not parse authored USD dependency locator {locator}: {exc}",
        ) from exc
    package_inner: str | None = None
    outer = authored_path
    if Ar.IsPackageRelativePath(outer):
        outer, package_inner = Ar.SplitPackageRelativePathOuter(outer)
    if not outer:
        return None
    file_uri_path = _canonical_local_file_uri_path(
        outer,
        code="artifact_dependency_preflight_failed",
        label="authored USD dependency",
    )
    if file_uri_path is not None:
        outer = str(file_uri_path)
    elif "://" in outer or _is_remote_resolver_locator(outer):
        return None

    def projected_identifier(path: Path) -> str:
        rebuilt = str(projection.projected_path(path))
        if package_inner is not None:
            rebuilt = Ar.JoinPackageRelativePath(rebuilt, package_inner)
        if arguments:
            rebuilt = Sdf.Layer.CreateIdentifier(rebuilt, arguments)
        return rebuilt

    # Authored USD locators are content, not shell input.  Keep ``~`` literal.
    authored_outer = Path(outer)
    absolute_authored = authored_outer.is_absolute()
    candidate = authored_outer
    if not absolute_authored:
        candidate = owner_path.parent / candidate
    candidate = Path(os.path.abspath(candidate))
    try:
        os.stat(candidate, follow_symlinks=False)
    except FileNotFoundError:
        if not absolute_authored:
            # Unanchored relative assets may be supplied by the active resolver
            # context (for example, a configured search root). Seal a local
            # absolute result immediately and rewrite composition to that copy.
            try:
                resolved_outer = str(Ar.GetResolver().Resolve(outer))
            except Exception as exc:
                raise JointRiggerContractError(
                    "artifact_dependency_preflight_failed",
                    f"could not resolve authored USD dependency {locator}: {exc}",
                ) from exc
            resolved_file_uri_path = _canonical_local_file_uri_path(
                resolved_outer,
                code="artifact_dependency_preflight_failed",
                label="resolved USD dependency",
            )
            if resolved_file_uri_path is not None:
                resolved_outer = str(resolved_file_uri_path)
            elif "://" in resolved_outer or _is_remote_resolver_locator(resolved_outer):
                return None
            if not resolved_outer:
                return _ProjectedLocator(
                    path=None,
                    format_hint=None,
                    inspect_layer=False,
                    rewritten=projected_identifier(candidate),
                )
            resolved_candidate = Path(resolved_outer)
            if not resolved_candidate.is_absolute():
                resolved_candidate = owner_path.parent / resolved_candidate
                return _ProjectedLocator(
                    path=None,
                    format_hint=None,
                    inspect_layer=False,
                    rewritten=projected_identifier(
                        Path(os.path.abspath(resolved_candidate))
                    ),
                )
            candidate = Path(os.path.abspath(resolved_candidate))
            _copy_precomposition_file(projection, candidate)
        else:
            # Redirect an absent absolute path into the private tree so a later
            # live pathname creation can never enter composition.
            return _ProjectedLocator(
                path=None,
                format_hint=None,
                inspect_layer=False,
                rewritten=projected_identifier(candidate),
            )
    except OSError as exc:
        raise JointRiggerContractError(
            "dependency_artifact_invalid",
            f"could not inspect authored local USD dependency {locator}: {exc}",
        ) from exc

    format_hint = str(arguments.get("format") or "") or None
    inspect_layer = (
        package_inner is None
        and _precomposition_layer_suffix(candidate, format_hint=format_hint) is not None
    )
    return _ProjectedLocator(
        path=candidate,
        format_hint=format_hint,
        inspect_layer=inspect_layer,
        rewritten=projected_identifier(candidate),
    )


def _is_remote_resolver_locator(locator: str) -> bool:
    """Keep opaque resolver identifiers out of the local projection mirror."""

    try:
        parsed = urlparse(locator)
    except ValueError:
        return True
    if not parsed.scheme or parsed.scheme == "file":
        return False
    # A rooted native Windows drive is a local spelling, even though urlparse
    # reports its one-letter drive prefix as a scheme. Other one-letter schemes
    # such as ``s:asset`` are resolver identifiers and must remain untouched.
    # Drive-relative ``C:asset`` spellings are intentionally not portable here.
    is_rooted_windows_drive = (
        len(parsed.scheme) == 1
        and len(locator) >= 3
        and locator[0].isalpha()
        and locator[1] == ":"
        and locator[2] in {"/", "\\"}
    )
    return not is_rooted_windows_drive


def _canonical_local_file_uri_path(
    locator: str,
    *,
    code: str,
    label: str,
) -> Path | None:
    """Decode one exact canonical absolute local file URI without resolving it."""

    if "\x00" in locator:
        _fail(code, f"{label} contains an embedded NUL")
    try:
        parsed = urlparse(locator)
    except ValueError:
        _fail(
            code,
            f"{label} must use an exact canonical absolute file URI: {locator}",
        )
    if parsed.scheme != "file":
        return None
    decoded_path = unquote(parsed.path)
    if "\x00" in decoded_path or decoded_path.startswith("//"):
        _fail(
            code,
            f"{label} must use an exact canonical absolute file URI: {locator}",
        )
    local_path = Path(decoded_path)
    try:
        canonical_uri = local_path.as_uri()
    except ValueError:
        _fail(
            code,
            f"{label} must use an exact canonical absolute file URI: {locator}",
        )
    has_dot_segment = any(segment in {".", ".."} for segment in decoded_path.split("/"))
    if (
        parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or has_dot_segment
        or locator != canonical_uri
    ):
        _fail(
            code,
            f"{label} must use an exact canonical absolute file URI: {locator}",
        )
    return local_path


def _require_cached_layer_matches_projection(
    path: Path,
    projected_layer: Any,
    *,
    format_hint: str | None,
    Sdf: Any,
) -> None:
    """Preserve stale/dirty process-global cache failures without live reopen."""

    identifier = str(path)
    if format_hint:
        identifier = Sdf.Layer.CreateIdentifier(identifier, {"format": format_hint})
    cached = Sdf.Layer.Find(identifier)
    if cached is None:
        return
    if bool(getattr(cached, "dirty", False)):
        _fail(
            "artifact_dependency_cache_dirty",
            f"refusing to read through unsaved edits in cached USD dependency layer: {identifier}",
        )
    try:
        cached_text = cached.ExportToString()
        projected_text = projected_layer.ExportToString()
    except Exception as exc:
        raise JointRiggerContractError(
            "artifact_dependency_refresh_failed",
            f"could not compare cached USD dependency layer {identifier} with "
            f"private projection: {exc}",
        ) from exc
    if bool(getattr(cached, "dirty", False)):
        _fail(
            "artifact_dependency_cache_dirty",
            f"cached USD dependency layer became dirty while comparing its private projection: {identifier}",
        )
    if cached_text != projected_text:
        _fail(
            "artifact_dependency_cache_stale",
            "cached USD dependency layer differs from disk; release cached stages "
            f"or retry in a fresh process: {identifier}",
        )


def _require_cached_package_matches_projection(
    path: Path,
    record: _ProjectedLocalFile,
    *,
    Sdf: Any,
) -> None:
    """Compare an already-cached package root with the private package copy."""

    cached = Sdf.Layer.Find(str(path))
    if cached is None:
        return
    try:
        projected = Sdf.Layer.OpenAsAnonymous(str(record.projected_path))
    except Exception as exc:
        raise JointRiggerContractError(
            "artifact_dependency_refresh_failed",
            f"could not read private USD package projection {path}: {exc}",
        ) from exc
    if projected is None:
        _fail(
            "artifact_dependency_refresh_failed",
            f"could not read private USD package projection: {path}",
        )
    _require_cached_layer_matches_projection(
        path,
        projected,
        format_hint=None,
        Sdf=Sdf,
    )


def _precomposition_descriptor_sha256(
    descriptor: int,
    *,
    size: int,
    label: str,
) -> str:
    """Hash exact bounded descriptor bytes without consuming descriptor state."""

    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(
            descriptor,
            min(_AR_ASSET_READ_CHUNK_SIZE, size - offset),
            offset,
        )
        if not chunk:
            _fail(
                "dependency_artifact_invalid",
                f"local USD dependency changed while hashing: {label}",
            )
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, offset):
        _fail(
            "dependency_artifact_invalid",
            f"local USD dependency grew while hashing: {label}",
        )
    return digest.hexdigest()


def _require_projected_root_matches_hash(
    path: Path,
    *,
    projection: _UsdCompositionProjection,
    expected_sha256: str,
    code: str,
) -> None:
    """Bind the projected root bytes to the caller's initial stable hash."""

    absolute = Path(os.path.abspath(path.expanduser()))
    record = projection.files.get(absolute)
    if record is None or record.sha256 != expected_sha256:
        _fail(code, f"USD root changed before private projection: {path}")


def _precomposition_layer_suffix(
    path: Path,
    *,
    format_hint: str | None,
) -> str | None:
    """Return the safe snapshot suffix for one inspectable USD layer."""

    if format_hint:
        normalized = format_hint.lower().lstrip(".")
        if normalized in _PRECOMPOSITION_LAYER_FORMATS:
            return f".{normalized}"
        return None
    suffix = path.suffix.lower()
    return suffix if suffix in _PRECOMPOSITION_LAYER_SUFFIXES else None


def _open_precomposition_regular_file(
    path: Path,
    *,
    locator: str | None = None,
) -> tuple[
    int,
    _PrecompositionFileState,
    Path,
    tuple[_PrecompositionSymlinkHop, ...],
]:
    """Open one local locator without following or blocking on special files."""

    backing_path, symlink_hops = _resolve_precomposition_symlinks(path)
    detail = (
        f"{locator} -> {path} -> {backing_path}"
        if locator is not None
        else f"{path} -> {backing_path}"
    )
    try:
        expected = os.stat(backing_path, follow_symlinks=False)
    except OSError as exc:
        raise JointRiggerContractError(
            "dependency_artifact_invalid",
            f"could not stat authored local USD dependency {detail}: {exc}",
        ) from exc
    if not stat.S_ISREG(expected.st_mode):
        _fail(
            "dependency_artifact_invalid",
            "authored local USD dependency is not a non-symlink regular file: "
            f"{detail}",
        )
    expected_state = _precomposition_file_state(expected)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(backing_path, flags)
    except OSError as exc:
        raise JointRiggerContractError(
            "dependency_artifact_invalid",
            f"could not open authored local USD dependency {detail}: {exc}",
        ) from exc
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        os.close(descriptor)
        raise JointRiggerContractError(
            "dependency_artifact_invalid",
            f"could not inspect opened local USD dependency {detail}: {exc}",
        ) from exc
    if (
        not stat.S_ISREG(opened.st_mode)
        or _precomposition_file_state(opened) != expected_state
    ):
        os.close(descriptor)
        _fail(
            "dependency_artifact_invalid",
            f"authored local USD dependency changed before inspection: {detail}",
        )
    return descriptor, expected_state, backing_path, symlink_hops


def _require_precomposition_file_unchanged(
    path: Path,
    *,
    descriptor: int,
    expected_state: _PrecompositionFileState,
    locator: str | None = None,
) -> None:
    """Require the descriptor and authored path to retain their exact state."""

    detail = f"{locator} -> {path}" if locator is not None else str(path)
    try:
        descriptor_state = _precomposition_file_state(os.fstat(descriptor))
        path_state = _precomposition_file_state(os.stat(path, follow_symlinks=False))
    except OSError as exc:
        raise JointRiggerContractError(
            "dependency_artifact_invalid",
            f"authored local USD dependency changed during inspection {detail}: {exc}",
        ) from exc
    if descriptor_state != expected_state or path_state != expected_state:
        _fail(
            "dependency_artifact_invalid",
            f"authored local USD dependency changed during inspection: {detail}",
        )


def _resolve_precomposition_symlinks(
    path: Path,
) -> tuple[Path, tuple[_PrecompositionSymlinkHop, ...]]:
    """Resolve lexical symlink hops without opening through any of them."""

    absolute = Path(os.path.abspath(path.expanduser()))
    current = Path(absolute.anchor)
    remaining = list(absolute.parts[1:])
    visited: set[Path] = set()
    hops: list[_PrecompositionSymlinkHop] = []
    for _ in range(_MAX_DEPENDENCY_VALIDATION_PASSES):
        if not remaining:
            return current, tuple(hops)
        current /= remaining.pop(0)
        try:
            metadata = os.stat(current, follow_symlinks=False)
        except OSError as exc:
            raise JointRiggerContractError(
                "dependency_artifact_invalid",
                f"could not inspect authored local USD dependency {path}: {exc}",
            ) from exc
        if not stat.S_ISLNK(metadata.st_mode):
            continue
        if current in visited:
            _fail(
                "dependency_artifact_invalid",
                f"authored local USD dependency contains a symlink cycle: {path}",
            )
        visited.add(current)
        expected_state = _precomposition_file_state(metadata)
        try:
            target = os.readlink(current)
            observed_state = _precomposition_file_state(
                os.stat(current, follow_symlinks=False)
            )
        except OSError as exc:
            raise JointRiggerContractError(
                "dependency_artifact_invalid",
                f"authored local USD dependency symlink changed: {current}: {exc}",
            ) from exc
        if observed_state != expected_state:
            _fail(
                "dependency_artifact_invalid",
                f"authored local USD dependency symlink changed: {current}",
            )
        hops.append((current, expected_state, target))
        rewritten = Path(target)
        if not rewritten.is_absolute():
            rewritten = current.parent / rewritten
        rewritten = Path(os.path.abspath(rewritten.joinpath(*remaining)))
        current = Path(rewritten.anchor)
        remaining = list(rewritten.parts[1:])
    _fail(
        "dependency_artifact_invalid",
        "authored local USD dependency symlink resolution did not converge after "
        f"{_MAX_DEPENDENCY_VALIDATION_PASSES} hops: {path}",
    )
    raise AssertionError("unreachable")


def _require_precomposition_symlinks_unchanged(
    hops: Sequence[_PrecompositionSymlinkHop],
    *,
    locator: str | None = None,
) -> None:
    """Require every lexical symlink hop to retain its inode and target."""

    for path, expected_state, expected_target in hops:
        try:
            observed_state = _precomposition_file_state(
                os.stat(path, follow_symlinks=False)
            )
            observed_target = os.readlink(path)
        except OSError as exc:
            raise JointRiggerContractError(
                "dependency_artifact_invalid",
                f"authored local USD dependency symlink changed {path}: {exc}",
            ) from exc
        if observed_state != expected_state or observed_target != expected_target:
            detail = f"{locator}: {path}" if locator is not None else str(path)
            _fail(
                "dependency_artifact_invalid",
                f"authored local USD dependency symlink changed: {detail}",
            )


def _precomposition_file_state(
    value: os.stat_result,
) -> _PrecompositionFileState:
    """Return the exact inode state used by precomposition validation."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_dependency_structure(
    path: Path,
    *,
    logical_artifact_path: Path,
) -> tuple[_CapturedDependencyStructure, ...]:
    """Capture canonical dependency structure without hashing live paths."""

    artifact = path.expanduser().resolve(strict=True)
    logical_artifact = Path(os.path.abspath(logical_artifact_path.expanduser()))
    records: list[_CapturedDependencyStructure] = []
    for dependency in _enumerate_usd_dependencies(artifact):
        is_root = (
            dependency.kind == "layer"
            and not dependency.package_relative
            and dependency.local_path == artifact
        )
        if is_root:
            records.append(
                _CapturedDependencyStructure(
                    kind="stage_root_layer",
                    locator="$artifact",
                    backing_path=None,
                    package_inner_locator=None,
                )
            )
            continue
        if dependency.local_path is None:
            _fail(
                "unbound_dependency_artifact",
                "a no-sidecar dependency has no bindable local backing file: "
                f"{dependency.identifier}",
            )
        assert dependency.local_path is not None
        backing_path = dependency.local_path.expanduser().resolve(strict=True)
        package_inner_locator = (
            dependency.identifier if dependency.package_relative else None
        )
        records.append(
            _CapturedDependencyStructure(
                kind=("used_layer" if dependency.kind == "layer" else dependency.kind),
                locator=_canonical_layer_locator(
                    dependency.identifier,
                    artifact_path=logical_artifact,
                ),
                backing_path=backing_path,
                package_inner_locator=package_inner_locator,
            )
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.kind,
                record.locator,
                str(record.backing_path or ""),
                record.package_inner_locator or "",
            ),
        )
    )


def _artifact_identity_from_captured_records(
    *,
    logical_artifact_path: Path,
    uri: str,
    root_sha256: str,
    records: Sequence[_CapturedDependencyIdentityRecord],
) -> ArtifactIdentityV1:
    """Build an identity from frozen records without touching the filesystem."""

    return _artifact_identity(
        logical_artifact_path,
        uri,
        None,
        root_sha256=root_sha256,
        dependency_records=tuple(record.identity_record() for record in records),
    )


def local_usd_dependency_paths(
    path: str | Path,
    *,
    include_lexical_aliases: bool = False,
) -> tuple[Path, ...]:
    """Return every resolved local file in a USD and opaque-material closure.

    Paths are absolute, symlink-resolved, deduplicated, and sorted by default.
    With ``include_lexical_aliases=True``, the absolute authored locator is also
    returned when it differs from its resolved filesystem referent.  This lets
    destructive publication preflight protect both names. Package entries map
    to their outermost local package file. Any unresolved authored dependency
    fails closed instead of returning a partial inventory.
    """

    artifact_path = Path(path)
    root_sha256 = _file_sha256(artifact_path, code="artifact_missing")
    with _usd_composition_projection((artifact_path,)) as projection:
        _require_projected_root_matches_hash(
            artifact_path,
            projection=projection,
            expected_sha256=root_sha256,
            code="artifact_mutated",
        )
        dependencies = _enumerate_usd_dependencies(
            artifact_path,
            projection=projection,
            root_mutated_code="artifact_mutated",
            dependency_mutated_code="artifact_dependency_mutated",
        )
        projection.require_unchanged(
            artifact_path,
            code="artifact_dependency_mutated",
            root_code="artifact_mutated",
        )
        paths = {
            item.local_path for item in dependencies if item.local_path is not None
        }
        resolved_artifact = artifact_path.expanduser().resolve(strict=False)
        paths.add(resolved_artifact)
        if include_lexical_aliases:
            for item in dependencies:
                if item.lexical_path is not None:
                    paths.update(_local_path_alias_chain(item.lexical_path))
            paths.add(Path(os.path.abspath(artifact_path.expanduser())))
        return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _local_path_alias_chain(path: Path) -> tuple[Path, ...]:
    """Return full dependency locators at every symlink resolution hop.

    Recording only the authored locator and final referent is insufficient for
    destructive-target preflight: an intermediate symlink can live inside a
    sidecar that publication replaces recursively.  Each returned path keeps
    the unresolved suffix after the encountered symlink, so ancestor-directory
    aliases are protected as well as leaf aliases.
    """

    absolute = Path(os.path.abspath(path.expanduser()))
    aliases = [absolute]
    current = Path(absolute.anchor)
    remaining = list(absolute.parts[1:])
    visited_symlinks: set[Path] = set()
    while remaining:
        current /= remaining.pop(0)
        if not current.is_symlink():
            continue
        if current in visited_symlinks:
            _fail(
                "dependency_artifact_invalid",
                f"USD dependency contains a symlink cycle: {current}",
            )
        visited_symlinks.add(current)
        target = current.readlink()
        if not target.is_absolute():
            target = current.parent / target
        rewritten = Path(os.path.abspath(target.joinpath(*remaining)))
        aliases.append(rewritten)
        current = Path(rewritten.anchor)
        remaining = list(rewritten.parts[1:])
    return tuple(aliases)


def extract_reference_input(
    source_usd_path: str | Path,
    rigged_reference_usd_path: str | Path,
    *,
    source_uri: str,
    reference_uri: str,
    joint_paths: Sequence[str] | None = None,
    allowed_omitted_joint_types: Iterable[str] = (),
) -> JointRiggerInputV1:
    """Export a shared authoring input from a paired rigged reference.

    ``source_uri`` and ``reference_uri`` are caller-owned stable identifiers;
    local paths are used only to open and hash the stages.  An explicit
    ``joint_paths`` selection is checked against every authored joint.  Omitted
    types must be acknowledged through ``allowed_omitted_joint_types``.
    """

    from pxr import Usd, UsdGeom, UsdPhysics

    source_path = Path(source_usd_path)
    reference_path = Path(rigged_reference_usd_path)
    source_before = _file_sha256(source_path, code="source_artifact_missing")
    reference_before = _file_sha256(
        reference_path,
        code="reference_artifact_missing",
    )
    with _usd_composition_projection((source_path, reference_path)) as projection:
        return _extract_reference_input_from_projection(
            source_path=source_path,
            reference_path=reference_path,
            source_uri=source_uri,
            reference_uri=reference_uri,
            joint_paths=joint_paths,
            allowed_omitted_joint_types=allowed_omitted_joint_types,
            source_before=source_before,
            reference_before=reference_before,
            projection=projection,
            Usd=Usd,
            UsdGeom=UsdGeom,
            UsdPhysics=UsdPhysics,
        )


def _extract_reference_input_from_projection(
    *,
    source_path: Path,
    reference_path: Path,
    source_uri: str,
    reference_uri: str,
    joint_paths: Sequence[str] | None,
    allowed_omitted_joint_types: Iterable[str],
    source_before: str,
    reference_before: str,
    projection: _UsdCompositionProjection,
    Usd: Any,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> JointRiggerInputV1:
    """Extract paired evidence while the complete private projection is alive."""

    _require_projected_root_matches_hash(
        source_path,
        projection=projection,
        expected_sha256=source_before,
        code="source_artifact_mutated",
    )
    _require_projected_root_matches_hash(
        reference_path,
        projection=projection,
        expected_sha256=reference_before,
        code="reference_artifact_mutated",
    )
    source_stage = _open_stage(
        projection.projected_path(source_path),
        Usd=Usd,
        label="source",
        display_path=source_path,
    )
    reference_stage = _open_stage(
        projection.projected_path(reference_path),
        Usd=Usd,
        label="reference",
        display_path=reference_path,
    )
    source_dependencies = _enumerate_usd_dependencies(
        source_path,
        projection=projection,
        root_mutated_code="source_artifact_mutated",
        dependency_mutated_code="source_dependency_artifact_mutated",
    )
    reference_dependencies = _enumerate_usd_dependencies(
        reference_path,
        projection=projection,
        root_mutated_code="reference_artifact_mutated",
        dependency_mutated_code="reference_dependency_artifact_mutated",
    )
    projection.require_unchanged(
        source_path,
        code="source_dependency_artifact_mutated",
        root_code="source_artifact_mutated",
    )
    projection.require_unchanged(
        reference_path,
        code="reference_dependency_artifact_mutated",
        root_code="reference_artifact_mutated",
    )
    _require_compatible_stage_frames(
        source_stage,
        reference_stage,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
    )

    source_identity = _artifact_identity(
        source_path,
        source_uri,
        source_stage,
        root_sha256=source_before,
        dependency_records=_dependency_identity_records(
            source_path,
            source_dependencies,
        ),
    )
    reference_identity = _artifact_identity(
        reference_path,
        reference_uri,
        reference_stage,
        root_sha256=reference_before,
        dependency_records=_dependency_identity_records(
            reference_path,
            reference_dependencies,
        ),
    )
    selected_prims = _select_joint_prims(
        reference_stage,
        joint_paths=joint_paths,
        allowed_omitted_joint_types=allowed_omitted_joint_types,
        UsdPhysics=UsdPhysics,
    )
    if not selected_prims:
        _fail("no_supported_joints", "reference selection contains no joints")
    _require_selected_joint_paths_absent_from_source(source_stage, selected_prims)

    source_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    reference_xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    plans = tuple(
        _extract_joint_plan(
            prim,
            source_stage=source_stage,
            reference_stage=reference_stage,
            reference_identity=reference_identity,
            source_xform_cache=source_xform_cache,
            xform_cache=reference_xform_cache,
            UsdPhysics=UsdPhysics,
            UsdGeom=UsdGeom,
        )
        for prim in selected_prims
    )
    body_paths = {
        endpoint
        for plan in plans
        for endpoint in (plan.topology.body0, plan.topology.body1)
    }
    _require_source_joints_replay_safe(
        source_stage,
        reference_stage=reference_stage,
        UsdPhysics=UsdPhysics,
    )
    _require_source_physics_subset(
        source_stage,
        reference_stage=reference_stage,
        body_paths=body_paths,
        UsdPhysics=UsdPhysics,
        UsdGeom=UsdGeom,
    )
    rigid_bodies = _extract_rigid_body_plans(
        reference_stage,
        source_stage=source_stage,
        body_paths=body_paths,
        reference_identity=reference_identity,
        UsdPhysics=UsdPhysics,
        UsdGeom=UsdGeom,
    )
    articulation_root = _extract_articulation_root(
        reference_stage,
        source_stage=source_stage,
        body_paths=body_paths,
        joint_paths={plan.topology.joint_id for plan in plans},
        reference_identity=reference_identity,
        UsdPhysics=UsdPhysics,
    )
    _require_unmodeled_physics_facts_preexisting(
        source_stage,
        reference_stage=reference_stage,
        body_paths=body_paths,
        rigid_bodies=rigid_bodies,
        articulation_root=articulation_root,
        UsdPhysics=UsdPhysics,
    )

    result = JointRiggerInputV1(
        schema_version=INPUT_SCHEMA_VERSION,
        source_asset=source_identity,
        plan=JointRiggerPlanV1(
            schema_version=PLAN_SCHEMA_VERSION,
            joints=plans,
            rigid_bodies=rigid_bodies,
            articulation_root=articulation_root,
        ),
    )
    _require_artifact_identity_unchanged(
        source_path,
        source_stage,
        source_identity,
        dependencies=source_dependencies,
        projection=projection,
        missing_code="source_artifact_missing",
        root_mutated_code="source_artifact_mutated",
        dependency_mutated_code="source_dependency_artifact_mutated",
    )
    _require_artifact_identity_unchanged(
        reference_path,
        reference_stage,
        reference_identity,
        dependencies=reference_dependencies,
        projection=projection,
        missing_code="reference_artifact_missing",
        root_mutated_code="reference_artifact_mutated",
        dependency_mutated_code="reference_dependency_artifact_mutated",
    )
    return result


def _require_selected_joint_paths_absent_from_source(
    source_stage: Any,
    selected_prims: Sequence[Any],
) -> None:
    """Require the paired source to be genuinely pre-authoring at selected paths."""

    present = sorted(
        str(prim.GetPath())
        for prim in selected_prims
        if source_stage.GetPrimAtPath(prim.GetPath()).IsValid()
    )
    if present:
        _fail(
            "selected_joint_present_in_source",
            "selected joint paths must be absent from the paired source; "
            f"found {present}",
        )


def write_reference_input(path: str | Path, value: JointRiggerInputV1) -> None:
    """Atomically write one canonical reference input without a trailing newline."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(canonical_json(value))
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _extract_joint_plan(
    prim: Any,
    *,
    source_stage: Any,
    reference_stage: Any,
    reference_identity: ArtifactIdentityV1,
    source_xform_cache: Any,
    xform_cache: Any,
    UsdPhysics: Any,
    UsdGeom: Any,
) -> JointPlanV1:
    joint_path = str(prim.GetPath())
    joint_type = _joint_type(prim, UsdPhysics)
    if joint_type not in _SUPPORTED_JOINT_TYPES:
        _fail(
            "unsupported_joint_type",
            f"{joint_path} uses unsupported joint type {joint_type!r}",
        )
    if joint_type == "spherical":
        axis_attribute = prim.GetAttribute("physics:axis")
        if _is_authored_value_only_attribute(
            axis_attribute,
            owner_path=joint_path,
            connection_code="unsupported_spherical_axis",
        ):
            _require_static_attribute(axis_attribute, owner_path=joint_path)
            _fail(
                "unsupported_spherical_axis",
                f"{joint_path} has unrepresented spherical axis opinion: physics:axis",
            )
        authored_rotations = [
            attribute.GetName()
            for index in (0, 1)
            if _is_authored_value_only_attribute(
                attribute := prim.GetAttribute(f"physics:localRot{index}"),
                owner_path=joint_path,
                connection_code="unsupported_spherical_orientation",
            )
        ]
        if authored_rotations:
            _fail(
                "unsupported_spherical_orientation",
                f"{joint_path} has unrepresented spherical frame orientation "
                f"opinions: {authored_rotations}",
            )
    _reject_unrepresented_base_joint_properties(prim, joint_path=joint_path)
    _reject_unrepresented_joint_schemas(prim, joint_path=joint_path)
    body0 = _single_relationship_target(prim, "physics:body0", endpoint="body0")
    body1 = _single_relationship_target(prim, "physics:body1", endpoint="body1")
    if body0 == body1:
        _fail("same_body_endpoints", f"{joint_path} targets {body0} twice")
    for endpoint, path in (("body0", body0), ("body1", body1)):
        _require_resolved_prim(reference_stage, path, endpoint, joint_path)
        _require_resolved_prim(source_stage, path, endpoint, joint_path, source=True)
        _require_matching_endpoint_world_transform(
            source_stage,
            reference_stage,
            path=path,
            endpoint=endpoint,
            joint_path=joint_path,
            source_xform_cache=source_xform_cache,
            reference_xform_cache=xform_cache,
            UsdGeom=UsdGeom,
        )

    type_provenance = _provenance(
        reference_identity,
        joint_path,
        ("typeName",),
        f"Authored {joint_type} schema on {joint_path}.",
    )
    field_provenance: dict[str, FieldProvenanceV1] = {
        "joint_type": type_provenance,
        "body0": _provenance(
            reference_identity,
            joint_path,
            ("physics:body0",),
            f"Authored body0 relationship on {joint_path}.",
        ),
        "body1": _provenance(
            reference_identity,
            joint_path,
            ("physics:body1",),
            f"Authored body1 relationship on {joint_path}.",
        ),
    }
    axis_stage: tuple[float, float, float] | None = None
    if joint_type in {"revolute", "prismatic"}:
        axis_stage, axis_properties = _stage_axis(
            prim,
            reference_stage=reference_stage,
            body0=body0,
            body1=body1,
            joint_path=joint_path,
            xform_cache=xform_cache,
        )
        field_provenance["axis_stage"] = FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=joint_path,
            properties=axis_properties,
            derivation="joint_local_axis_to_stage_frame",
            evidence=f"USD joint frames establish the signed stage axis for {joint_path}.",
        )

    topology = JointTopologyV1(
        joint_id=joint_path,
        joint_type=cast(Literal["revolute", "prismatic", "spherical"], joint_type),
        body0=body0,
        body1=body1,
        axis_stage=axis_stage,
        field_provenance=field_provenance,
    )
    drive_instances, authored_drive_properties = _drive_schema_context(
        prim,
        joint_path=joint_path,
    )
    max_joint_velocity, physx_drive_properties, joint_friction = (
        _extract_physx_joint_opinions(
            prim,
            joint_type=joint_type,
            joint_path=joint_path,
            has_drive=bool(drive_instances),
            reference_identity=reference_identity,
        )
    )
    return JointPlanV1(
        topology=topology,
        limit=_extract_limit(
            prim,
            joint_type=joint_type,
            joint_path=joint_path,
            reference_stage=reference_stage,
            reference_identity=reference_identity,
            UsdGeom=UsdGeom,
        ),
        anchor=_extract_anchor(
            prim,
            reference_stage=reference_stage,
            body0=body0,
            body1=body1,
            joint_path=joint_path,
            reference_identity=reference_identity,
            xform_cache=xform_cache,
        ),
        joint_friction=joint_friction,
        drive=_extract_drive(
            prim,
            joint_type=joint_type,
            joint_path=joint_path,
            reference_identity=reference_identity,
            UsdPhysics=UsdPhysics,
            instances=drive_instances,
            authored_properties=authored_drive_properties,
            max_joint_velocity=max_joint_velocity,
            physx_properties=physx_drive_properties,
        ),
    )


def _select_joint_prims(
    stage: Any,
    *,
    joint_paths: Sequence[str] | None,
    allowed_omitted_joint_types: Iterable[str],
    UsdPhysics: Any,
) -> tuple[Any, ...]:
    joint_candidates = {
        str(prim.GetPath()): prim
        for prim in _traverse_all_prims(stage)
        if prim.IsA(UsdPhysics.Joint)
    }
    instance_proxy_joints = sorted(
        path
        for path, prim in joint_candidates.items()
        if _is_active_defined_prim(prim) and prim.IsInstanceProxy()
    )
    if instance_proxy_joints:
        _fail(
            "unsupported_instance_proxy_physics",
            "reference joints are instance proxies and cannot be replayed without "
            f"reshaping: {instance_proxy_joints}",
        )
    all_joints = {
        path: prim
        for path, prim in joint_candidates.items()
        if _is_active_defined_prim(prim)
    }
    allowed = frozenset(
        str(value).strip().lower() for value in allowed_omitted_joint_types
    )
    if any(not value for value in allowed):
        _fail("invalid_joint_selection", "allowed omitted types must not be blank")
    if joint_paths is None:
        requested = tuple(sorted(all_joints))
    else:
        requested = tuple(joint_paths)
        if len(requested) != len(set(requested)):
            _fail("duplicate_joint_selection", "joint_paths contains duplicates")
        inactive = sorted(
            path
            for path in requested
            if path in joint_candidates and not joint_candidates[path].IsActive()
        )
        if inactive:
            _fail(
                "selected_joint_inactive",
                f"explicitly selected joints are inactive: {inactive}",
            )
        undefined = sorted(
            path
            for path in requested
            if path in joint_candidates and not joint_candidates[path].IsDefined()
        )
        if undefined:
            _fail(
                "selected_joint_undefined",
                f"explicitly selected joints are undefined: {undefined}",
            )
        missing = sorted(set(requested) - set(all_joints))
        if missing:
            _fail("selected_joint_missing", f"joint paths not found: {missing}")

    requested_set = set(requested)
    for path, prim in sorted(all_joints.items()):
        joint_type = _joint_type(prim, UsdPhysics)
        if path in requested_set:
            if joint_type not in _SUPPORTED_JOINT_TYPES:
                _fail(
                    "unsupported_joint_type",
                    f"selected joint {path} uses {joint_type!r}",
                )
            continue
        if joint_type not in allowed:
            _fail(
                "unapproved_omitted_joint",
                f"{path} ({joint_type}) was omitted without an explicit allowance",
            )
    return tuple(all_joints[path] for path in sorted(requested_set))


def _is_active_defined_prim(prim: Any) -> bool:
    return bool(prim and prim.IsValid() and prim.IsActive() and prim.IsDefined())


def _traverse_all_prims(stage: Any) -> Iterable[Any]:
    """Traverse all composed prims, including read-only instance proxies."""

    from pxr import Usd

    return cast(
        Iterable[Any],
        Usd.PrimRange.Stage(
            stage,
            Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ),
    )


def _joint_type(prim: Any, UsdPhysics: Any) -> str:
    schemas = (
        ("RevoluteJoint", "revolute"),
        ("PrismaticJoint", "prismatic"),
        ("SphericalJoint", "spherical"),
        ("FixedJoint", "fixed"),
        ("DistanceJoint", "distance"),
        ("D6Joint", "d6"),
    )
    for schema_name, value in schemas:
        schema = getattr(UsdPhysics, schema_name, None)
        if schema is not None and prim.IsA(schema):
            return value
    type_name = str(prim.GetTypeName()).removeprefix("Physics").removesuffix("Joint")
    return type_name.strip("_:- ").lower() or "unknown"


def _single_relationship_target(
    prim: Any,
    relationship_name: str,
    *,
    endpoint: str,
) -> str:
    targets = prim.GetRelationship(relationship_name).GetTargets()
    joint_path = str(prim.GetPath())
    if not targets:
        _fail(f"missing_{endpoint}", f"{joint_path} has no {relationship_name} target")
    if len(targets) != 1:
        _fail(
            f"multiple_{endpoint}_targets",
            f"{joint_path} has {len(targets)} {relationship_name} targets",
        )
    path = targets[0]
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.IsAbsoluteRootPath():
        _fail(
            f"invalid_{endpoint}_path",
            f"{joint_path} has invalid endpoint {path}",
        )
    return str(path)


def _require_resolved_prim(
    stage: Any,
    path: str,
    endpoint: str,
    joint_path: str,
    *,
    source: bool = False,
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not _is_active_defined_prim(prim):
        code = "endpoint_not_in_source" if source else "endpoint_not_in_reference"
        _fail(code, f"{joint_path} {endpoint} path does not resolve: {path}")
    if prim.IsInstanceProxy():
        stage_label = "source" if source else "reference"
        _fail(
            "unsupported_instance_proxy_physics",
            f"{joint_path} {endpoint} is an instance proxy in the "
            f"{stage_label} stage and cannot be targeted without reshaping: {path}",
        )


def _require_compatible_stage_frames(
    source_stage: Any,
    reference_stage: Any,
    *,
    UsdGeom: Any,
    UsdPhysics: Any,
) -> None:
    """Require paired stages to use the same effective frame and unit metadata."""

    source_meters = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    reference_meters = float(UsdGeom.GetStageMetersPerUnit(reference_stage))
    source_kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(source_stage))
    reference_kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(reference_stage))
    for label, meters, kilograms in (
        ("source", source_meters, source_kilograms),
        ("reference", reference_meters, reference_kilograms),
    ):
        if not all(
            math.isfinite(value) and value > 0.0 for value in (meters, kilograms)
        ):
            _fail(
                "invalid_stage_units",
                f"{label} stage units must be positive and finite; "
                f"metersPerUnit={meters!r}, kilogramsPerUnit={kilograms!r}",
            )
    source_up_axis = str(UsdGeom.GetStageUpAxis(source_stage))
    reference_up_axis = str(UsdGeom.GetStageUpAxis(reference_stage))
    if (
        source_up_axis != reference_up_axis
        or not math.isclose(
            source_meters,
            reference_meters,
            rel_tol=_STAGE_METADATA_TOLERANCE,
            abs_tol=_STAGE_METADATA_TOLERANCE,
        )
        or not math.isclose(
            source_kilograms,
            reference_kilograms,
            rel_tol=_STAGE_METADATA_TOLERANCE,
            abs_tol=_STAGE_METADATA_TOLERANCE,
        )
    ):
        _fail(
            "stage_metadata_mismatch",
            "source/reference stage metadata differs: "
            f"upAxis={source_up_axis!r}/{reference_up_axis!r}, "
            f"metersPerUnit={source_meters!r}/{reference_meters!r}, "
            f"kilogramsPerUnit={source_kilograms!r}/{reference_kilograms!r}",
        )


def _require_matching_endpoint_world_transform(
    source_stage: Any,
    reference_stage: Any,
    *,
    path: str,
    endpoint: str,
    joint_path: str,
    source_xform_cache: Any,
    reference_xform_cache: Any,
    UsdGeom: Any,
) -> None:
    """Reject paired endpoints whose default-time world frames differ."""

    _require_static_endpoint_transform(
        source_stage,
        path=path,
        endpoint=endpoint,
        joint_path=joint_path,
        stage_label="source",
        UsdGeom=UsdGeom,
    )
    _require_static_endpoint_transform(
        reference_stage,
        path=path,
        endpoint=endpoint,
        joint_path=joint_path,
        stage_label="reference",
        UsdGeom=UsdGeom,
    )
    source_matrix = source_xform_cache.GetLocalToWorldTransform(
        source_stage.GetPrimAtPath(path)
    )
    reference_matrix = reference_xform_cache.GetLocalToWorldTransform(
        reference_stage.GetPrimAtPath(path)
    )
    source_values = _matrix_values(source_matrix)
    reference_values = _matrix_values(reference_matrix)
    if not all(
        math.isclose(
            source_value,
            reference_value,
            rel_tol=0.0,
            abs_tol=_FRAME_TOLERANCE,
        )
        for source_value, reference_value in zip(
            source_values,
            reference_values,
            strict=True,
        )
    ):
        _fail(
            "endpoint_transform_mismatch",
            f"{joint_path} {endpoint} world transform differs between paired "
            f"stages: {path}",
        )


def _require_static_endpoint_transform(
    stage: Any,
    *,
    path: str,
    endpoint: str,
    joint_path: str,
    stage_label: str,
    UsdGeom: Any,
) -> None:
    """Reject time samples anywhere in a selected endpoint's transform chain."""

    prim = stage.GetPrimAtPath(path)
    while prim.IsValid() and not prim.IsPseudoRoot():
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            time_samples = sorted(
                {
                    float(time)
                    for op in xformable.GetOrderedXformOps()
                    for time in op.GetAttr().GetTimeSamples()
                }
            )
            if time_samples:
                _fail(
                    "time_varying_endpoint_transform",
                    f"{joint_path} {endpoint} has a time-sampled {stage_label} "
                    f"transform at {prim.GetPath()}: {time_samples}",
                )
        prim = prim.GetParent()


def _reject_unrepresented_base_joint_properties(
    prim: Any,
    *,
    joint_path: str,
) -> None:
    """Fail when authored base-Joint behavior cannot be represented by v1."""

    _reject_unrepresented_relationships(
        prim,
        owner_path=joint_path,
        relationship_names=("proxyPrim",),
        unsupported_code="unsupported_joint_relationship",
        schema_label="Joint",
    )
    for name, fallback in _BASE_JOINT_FALLBACK_PROPERTIES.items():
        attribute = prim.GetAttribute(name)
        if not _is_authored_value_only_attribute(
            attribute,
            owner_path=joint_path,
        ):
            continue
        _require_static_attribute(attribute, owner_path=joint_path)
        value = attribute.Get()
        if not isinstance(value, bool) or value is not fallback:
            _fail(
                "unsupported_joint_property",
                f"{joint_path} has nondefault {name} value {value!r}; "
                f"expected {fallback!r}",
            )
    for name in _UNREPRESENTED_JOINT_BREAK_PROPERTIES:
        attribute = prim.GetAttribute(name)
        if not _is_authored_value_only_attribute(
            attribute,
            owner_path=joint_path,
        ):
            continue
        _require_static_attribute(attribute, owner_path=joint_path)
        value = _optional_number(attribute.Get())
        if value is None or not math.isfinite(value):
            if value == math.inf:
                continue
            _fail(
                "invalid_joint_property",
                f"{joint_path} has invalid {name} value {attribute.Get()!r}",
            )
        _fail(
            "unsupported_joint_property",
            f"{joint_path} has an unrepresented finite {name} threshold: {value!r}",
        )


def _require_static_attribute(attribute: Any, *, owner_path: str) -> None:
    """Reject composed or raw temporal evidence for one static v1 property."""

    attribute_name = str(attribute.GetName())
    try:
        has_connections = bool(attribute.HasAuthoredConnections())
        connections = (
            tuple(str(path) for path in attribute.GetConnections())
            if has_connections
            else ()
        )
        time_samples = tuple(sorted(float(time) for time in attribute.GetTimeSamples()))
        has_spline_getter = getattr(attribute, "HasSpline", None)
        has_spline = bool(has_spline_getter()) if callable(has_spline_getter) else False
        might_vary_getter = getattr(attribute, "ValueMightBeTimeVarying", None)
        might_vary = bool(might_vary_getter()) if callable(might_vary_getter) else False
        raw_time_samples: list[str] = []
        raw_splines: list[str] = []
        raw_connections: list[str] = []
        property_stack_getter = getattr(attribute, "GetPropertyStack", None)
        property_stack = (
            tuple(property_stack_getter()) if callable(property_stack_getter) else ()
        )
        for property_spec in property_stack:
            layer = getattr(property_spec, "layer", None)
            path = getattr(property_spec, "path", None)
            source = f"{getattr(layer, 'identifier', '<unknown>')}:{path}"
            list_info_keys = getattr(property_spec, "ListInfoKeys", None)
            info_keys = (
                {str(key) for key in list_info_keys()}
                if callable(list_info_keys)
                else set()
            )
            if "spline" in info_keys:
                raw_splines.append(source)
            if "connectionPaths" in info_keys:
                raw_connections.append(source)
            list_time_samples = getattr(layer, "ListTimeSamplesForPath", None)
            if path is not None and callable(list_time_samples):
                samples = tuple(sorted(float(time) for time in list_time_samples(path)))
                if samples:
                    raw_time_samples.append(f"{source}={samples!r}")
        value_clip_sources = _ancestor_value_clip_sources(attribute)
    except Exception as exc:
        raise JointRiggerContractError(
            "static_property_time_unresolved",
            f"{owner_path} cannot prove {attribute_name} is static: "
            f"{type(exc).__name__}: {exc}",
        ) from exc

    if has_connections or raw_connections:
        _fail(
            "unsupported_attribute_connection",
            f"{owner_path} has authored connection opinions on value-only "
            f"{attribute_name}: composed={connections!r}, "
            f"raw={tuple(sorted(raw_connections))!r}",
        )
    if (
        time_samples
        or has_spline
        or might_vary
        or raw_time_samples
        or raw_splines
        or value_clip_sources
    ):
        _fail(
            "time_sampled_static_property",
            f"{owner_path} has non-static {attribute_name}: "
            f"samples={time_samples!r}, spline={has_spline}, "
            f"value_might_vary={might_vary}, "
            f"raw_time_samples={tuple(sorted(raw_time_samples))!r}, "
            f"raw_splines={tuple(sorted(raw_splines))!r}, "
            f"value_clip_sources={value_clip_sources!r}",
        )


def _ancestor_value_clip_sources(attribute: Any) -> tuple[str, ...]:
    """Return raw ancestor clip opinions that could drive this attribute."""

    get_prim = getattr(attribute, "GetPrim", None)
    if not callable(get_prim):
        return ()
    current = get_prim()
    sources: set[str] = set()
    while current and current.IsValid() and not current.IsPseudoRoot():
        for prim_spec in current.GetPrimStack():
            list_info_keys = getattr(prim_spec, "ListInfoKeys", None)
            info_keys = (
                {str(key) for key in list_info_keys()}
                if callable(list_info_keys)
                else set()
            )
            if "clips" not in info_keys:
                continue
            layer = getattr(prim_spec, "layer", None)
            sources.add(
                f"{getattr(layer, 'identifier', '<unknown>')}:"
                f"{getattr(prim_spec, 'path', '<unknown>')}"
            )
        current = current.GetParent()
    return tuple(sorted(sources))


def _is_authored_value_only_attribute(
    attribute: Any,
    *,
    owner_path: str,
    connection_code: str = "unsupported_attribute_connection",
) -> bool:
    """Return authored-value presence, rejecting unrepresented connections."""

    if attribute.HasAuthoredConnections():
        connections = [str(path) for path in attribute.GetConnections()]
        _fail(
            connection_code,
            f"{owner_path} has authored connection opinions on value-only "
            f"{attribute.GetName()}: {connections}",
        )
    return bool(attribute.HasAuthoredValueOpinion())


def _require_unrepresented_property_fallbacks(
    prim: Any,
    *,
    owner_path: str,
    properties: dict[str, bool | tuple[float, float, float]],
    unsupported_code: str,
    schema_label: str,
) -> None:
    """Allow explicit static fallbacks for properties omitted from the v1 model."""

    for name, fallback in properties.items():
        attribute = prim.GetAttribute(name)
        if not _is_authored_value_only_attribute(
            attribute,
            owner_path=owner_path,
        ):
            continue
        _require_static_attribute(attribute, owner_path=owner_path)
        value = attribute.Get()
        if not _matches_schema_fallback(value, fallback):
            _fail(
                unsupported_code,
                f"{owner_path} has nondefault {schema_label} {name} value "
                f"{value!r}; expected {fallback!r}",
            )


def _reject_unrepresented_relationships(
    prim: Any,
    *,
    owner_path: str,
    relationship_names: tuple[str, ...],
    unsupported_code: str,
    schema_label: str,
) -> None:
    """Reject authored schema relationships that the v1 plan cannot replay."""

    authored = []
    for name in relationship_names:
        relationship = prim.GetRelationship(name)
        if relationship and relationship.IsAuthored():
            authored.append(name)
    if authored:
        _fail(
            unsupported_code,
            f"{owner_path} has unrepresented {schema_label} relationships: "
            f"{sorted(authored)}",
        )


def _matches_schema_fallback(
    value: Any,
    fallback: bool | tuple[float, float, float],
) -> bool:
    if isinstance(fallback, bool):
        return isinstance(value, bool) and value is fallback
    try:
        components = tuple(float(component) for component in value)
    except (OverflowError, TypeError, ValueError):
        return False
    return components == fallback


def _stage_axis(
    prim: Any,
    *,
    reference_stage: Any,
    body0: str,
    body1: str,
    joint_path: str,
    xform_cache: Any,
) -> tuple[tuple[float, float, float], tuple[str, ...]]:
    from pxr import Gf

    axis_attribute = prim.GetAttribute("physics:axis")
    if not _is_authored_value_only_attribute(
        axis_attribute,
        owner_path=joint_path,
    ):
        _fail(
            "axis_unresolved",
            f"{joint_path} has no authored physics:axis value",
        )
    _require_static_attribute(axis_attribute, owner_path=joint_path)
    raw_axis = axis_attribute.Get()
    token = str(raw_axis).strip().lower() if raw_axis is not None else ""
    frame_bases = _AXIS_FRAME_BASES.get(token)
    if frame_bases is None:
        _fail("axis_unresolved", f"{joint_path} has unsupported axis {raw_axis!r}")
    assert frame_bases is not None

    frame_directions = []
    properties = [axis_attribute.GetName()]
    for index, body_path in enumerate((body0, body1)):
        body = reference_stage.GetPrimAtPath(body_path)
        local_rotation_attribute = prim.GetAttribute(f"physics:localRot{index}")
        local_rotation = None
        if _is_authored_value_only_attribute(
            local_rotation_attribute,
            owner_path=joint_path,
        ):
            _require_static_attribute(
                local_rotation_attribute,
                owner_path=joint_path,
            )
            local_rotation = local_rotation_attribute.Get()
            if local_rotation is None:
                _fail(
                    "axis_unresolved",
                    f"{joint_path} localRot{index} has no effective value",
                )
            properties.append(local_rotation_attribute.GetName())
        rotation = (
            _validated_joint_frame_rotation(
                local_rotation,
                joint_path=joint_path,
                field_name=f"physics:localRot{index}",
                Gf=Gf,
            )
            if local_rotation is not None
            else None
        )
        world_transform = xform_cache.GetLocalToWorldTransform(body)
        directions = []
        for basis in frame_bases:
            vector = Gf.Vec3d(*basis)
            if rotation is not None:
                vector = rotation.TransformDir(vector)
            vector = world_transform.TransformDir(vector)
            directions.append(_normalized_vector(vector, joint_path=joint_path))
        frame_directions.append(tuple(directions))
    frame_axes = (frame_directions[0][0], frame_directions[1][0])
    if _dot(frame_axes[0], frame_axes[1]) < 1.0 - _FRAME_TOLERANCE:
        _fail(
            "contradictory_joint_frames",
            f"{joint_path} body frames establish different signed axes",
        )
    if any(
        _dot(frame_directions[0][index], frame_directions[1][index])
        < 1.0 - _FRAME_TOLERANCE
        for index in (1, 2)
    ):
        _fail(
            "unsupported_joint_frame_twist",
            f"{joint_path} local joint frames differ around their shared axis",
        )
    return _canonical_vector(frame_axes[0]), tuple(properties)


def _extract_limit(
    prim: Any,
    *,
    joint_type: str,
    joint_path: str,
    reference_stage: Any,
    reference_identity: ArtifactIdentityV1,
    UsdGeom: Any,
) -> JointLimitV1 | None:
    lower_attr = prim.GetAttribute("physics:lowerLimit")
    upper_attr = prim.GetAttribute("physics:upperLimit")
    spherical_attrs = tuple(
        (name, prim.GetAttribute(name)) for name in _SPHERICAL_LIMIT_PROPERTIES
    )
    authored = {
        "physics:lowerLimit": _is_authored_value_only_attribute(
            lower_attr,
            owner_path=joint_path,
        ),
        "physics:upperLimit": _is_authored_value_only_attribute(
            upper_attr,
            owner_path=joint_path,
        ),
    }
    spherical_authored = {
        name: _is_authored_value_only_attribute(
            attribute,
            owner_path=joint_path,
        )
        for name, attribute in spherical_attrs
    }
    for attribute, present in (
        (lower_attr, authored["physics:lowerLimit"]),
        (upper_attr, authored["physics:upperLimit"]),
        *((attribute, spherical_authored[name]) for name, attribute in spherical_attrs),
    ):
        if present:
            _require_static_attribute(attribute, owner_path=joint_path)
    unsupported_spherical = [
        name for name, present in spherical_authored.items() if present
    ]
    if joint_type == "spherical":
        unsupported_spherical.extend(
            name for name, present in authored.items() if present
        )
    if unsupported_spherical:
        _fail(
            "unsupported_spherical_limit",
            f"{joint_path} has unrepresented spherical limit opinions: "
            f"{sorted(unsupported_spherical)}",
        )
    if not any(authored.values()):
        return None
    missing_effective = [
        name
        for name, attribute in (
            ("physics:lowerLimit", lower_attr),
            ("physics:upperLimit", upper_attr),
        )
        if authored[name] and attribute.Get() is None
    ]
    if missing_effective:
        _fail(
            "incomplete_optional_schema",
            f"{joint_path} limits have no effective value for {missing_effective}",
        )
    lower = _optional_number(
        lower_attr.Get() if authored["physics:lowerLimit"] else None
    )
    upper = _optional_number(
        upper_attr.Get() if authored["physics:upperLimit"] else None
    )
    if any(value is not None and not math.isfinite(value) for value in (lower, upper)):
        _fail("invalid_limit_value", f"{joint_path} has a non-finite limit")
    if lower is not None and upper is not None and lower > upper:
        _fail("invalid_limit_range", f"{joint_path} lower limit exceeds upper limit")
    if joint_type == "prismatic":
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(reference_stage))
        lower = None if lower is None else lower * meters_per_unit
        upper = None if upper is None else upper * meters_per_unit
        unit = "meters"
        derivation = "stage_units_to_meters"
    else:
        unit = "degrees"
        derivation = "usd_revolute_degrees"
    return JointLimitV1(
        lower=lower,
        upper=upper,
        unit=cast(Literal["degrees", "meters"], unit),
        provenance=FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=joint_path,
            properties=tuple(name for name, present in authored.items() if present),
            derivation=derivation,
            evidence=f"Authored USD limit opinions on {joint_path}.",
        ),
    )


def _extract_anchor(
    prim: Any,
    *,
    reference_stage: Any,
    body0: str,
    body1: str,
    joint_path: str,
    reference_identity: ArtifactIdentityV1,
    xform_cache: Any,
) -> JointAnchorV1 | None:
    from pxr import Gf

    attrs = [
        prim.GetAttribute("physics:localPos0"),
        prim.GetAttribute("physics:localPos1"),
    ]
    authored = [
        _is_authored_value_only_attribute(attribute, owner_path=joint_path)
        for attribute in attrs
    ]
    if not any(authored):
        return None
    for attribute, present in zip(attrs, authored, strict=True):
        if present:
            _require_static_attribute(attribute, owner_path=joint_path)
    if not all(authored):
        missing = [
            attribute.GetName()
            for attribute, present in zip(attrs, authored, strict=True)
            if not present
        ]
        _fail(
            "incomplete_optional_schema",
            f"{joint_path} anchor is missing authored fields: {missing}",
        )
    positions = []
    for index, (attribute, body_path) in enumerate(
        zip(attrs, (body0, body1), strict=True)
    ):
        value = attribute.Get()
        if value is None:
            _fail(
                "incomplete_optional_schema",
                f"{joint_path} localPos{index} has no effective value",
            )
        local_position = _finite_vector3(
            value,
            code="invalid_anchor_value",
            detail=f"{joint_path} localPos{index} contains non-finite values",
        )
        body = reference_stage.GetPrimAtPath(body_path)
        transformed = xform_cache.GetLocalToWorldTransform(body).Transform(
            Gf.Vec3d(*local_position)
        )
        positions.append(
            _finite_vector3(
                transformed,
                code="invalid_anchor_value",
                detail=(
                    f"{joint_path} localPos{index} transforms to a non-finite "
                    f"stage position through {body_path}"
                ),
            )
        )
    if _distance(positions[0], positions[1]) > _FRAME_TOLERANCE:
        _fail(
            "contradictory_joint_frames",
            f"{joint_path} localPos0/localPos1 do not establish one anchor",
        )
    return JointAnchorV1(
        position_stage=_canonical_vector(positions[0]),
        provenance=FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=joint_path,
            properties=tuple(attribute.GetName() for attribute in attrs),
            derivation="joint_local_positions_to_stage_frame",
            evidence=f"Authored joint anchor opinions on {joint_path}.",
        ),
    )


def _drive_schema_context(
    prim: Any,
    *,
    joint_path: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return one validated drive API/property context before PhysX extraction."""

    schema_tokens = _applied_schema_tokens(prim)
    instances = tuple(
        sorted(
            token.split(":", maxsplit=1)[1]
            for token in schema_tokens
            if token.startswith("PhysicsDriveAPI:")
        )
    )
    authored_properties = _authored_drive_properties(prim)
    for name in authored_properties:
        instance = _drive_property_instance(name)
        if instance is None or f"PhysicsDriveAPI:{instance}" not in schema_tokens:
            _fail(
                "drive_property_without_api",
                f"{joint_path} has authored drive property without its matching "
                f"PhysicsDriveAPI instance: {name}",
            )
    return instances, authored_properties


def _extract_drive(
    prim: Any,
    *,
    joint_type: str,
    joint_path: str,
    reference_identity: ArtifactIdentityV1,
    UsdPhysics: Any,
    instances: tuple[str, ...],
    authored_properties: tuple[str, ...],
    max_joint_velocity: float | None,
    physx_properties: tuple[str, ...],
) -> JointDriveV1 | None:
    if not instances:
        return None
    if len(instances) != 1:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} has multiple drive instances: {instances}",
        )
    instance = instances[0]
    expected_instance = {
        "revolute": "angular",
        "prismatic": "linear",
    }.get(joint_type)
    if expected_instance is None:
        _fail(
            "unsupported_drive_instance",
            f"{joint_path} cannot represent a drive for {joint_type!r} in v1",
        )
    if instance != expected_instance:
        _fail(
            "unsupported_drive_instance",
            f"{joint_path} {joint_type} drive must use {expected_instance!r}; "
            f"got {instance!r}",
        )
    unexpected_properties = sorted(
        name
        for name in authored_properties
        if name
        not in {
            f"drive:{instance}:physics:{suffix}" for suffix in _DRIVE_PROPERTY_SUFFIXES
        }
    )
    if unexpected_properties:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} has unrepresented drive opinions: {unexpected_properties}",
        )
    drive = UsdPhysics.DriveAPI.Get(prim, instance)
    getters = {
        "drive_type": drive.GetTypeAttr,
        "stiffness": drive.GetStiffnessAttr,
        "damping": drive.GetDampingAttr,
        "max_force": drive.GetMaxForceAttr,
        "target_position": drive.GetTargetPositionAttr,
        "target_velocity": drive.GetTargetVelocityAttr,
    }
    values: dict[str, Any] = {}
    missing = []
    properties = []
    for field, getter in getters.items():
        attribute = getter()
        if not _is_authored_value_only_attribute(
            attribute,
            owner_path=joint_path,
        ):
            missing.append(attribute.GetName())
            continue
        _require_static_attribute(attribute, owner_path=joint_path)
        value = attribute.Get()
        if value is None:
            missing.append(attribute.GetName())
            continue
        values[field] = value
        properties.append(attribute.GetName())
    if missing:
        _fail(
            "incomplete_optional_schema",
            f"{joint_path} drive is missing authored fields: {sorted(missing)}",
        )
    drive_type = str(values["drive_type"])
    if drive_type not in {"force", "acceleration"}:
        _fail("invalid_drive_type", f"{joint_path} uses drive type {drive_type!r}")
    numeric_values = {
        field: _required_finite_number(
            values[field],
            code="invalid_drive_value",
            detail=f"{joint_path} has invalid {field} drive value {values[field]!r}",
        )
        for field in (
            "stiffness",
            "damping",
            "max_force",
            "target_position",
            "target_velocity",
        )
    }
    for field in ("stiffness", "damping", "max_force"):
        if numeric_values[field] < 0.0:
            _fail(
                "invalid_drive_value",
                f"{joint_path} has negative {field} drive value "
                f"{numeric_values[field]!r}",
            )
    properties.extend(physx_properties)
    return JointDriveV1(
        drive_type=cast(Literal["force", "acceleration"], drive_type),
        stiffness=numeric_values["stiffness"],
        damping=numeric_values["damping"],
        max_force=numeric_values["max_force"],
        target_position=numeric_values["target_position"],
        target_velocity=numeric_values["target_velocity"],
        max_joint_velocity=max_joint_velocity,
        provenance=FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=joint_path,
            properties=tuple(properties),
            evidence=f"Complete authored {instance} drive on {joint_path}.",
        ),
    )


def _authored_drive_properties(prim: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(prop.GetName())
            for prop in prim.GetAuthoredProperties()
            if str(prop.GetName()).startswith(_DRIVE_PROPERTY_PREFIX)
        )
    )


def _drive_property_instance(name: str) -> str | None:
    parts = name.split(":")
    if (
        len(parts) != 4
        or parts[0] != "drive"
        or not parts[1]
        or parts[2] != "physics"
        or not parts[3]
    ):
        return None
    return parts[1]


def _extract_physx_joint_opinions(
    prim: Any,
    *,
    joint_type: str,
    joint_path: str,
    has_drive: bool,
    reference_identity: ArtifactIdentityV1,
) -> tuple[float | None, tuple[str, ...], JointFrictionV1 | None]:
    """Extract the represented PhysxJointAPI opinions or fail closed."""

    schema_tokens = _applied_schema_tokens(prim)
    physx_schemas = sorted(
        token
        for token in schema_tokens
        if token == _PHYSX_JOINT_SCHEMA or token.startswith(f"{_PHYSX_JOINT_SCHEMA}:")
    )
    if any(token != _PHYSX_JOINT_SCHEMA for token in physx_schemas):
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} uses unsupported PhysxJointAPI schemas: {physx_schemas}",
        )
    authored_properties = sorted(
        str(prop.GetName())
        for prop in prim.GetAuthoredProperties()
        if str(prop.GetName()).startswith("physxJoint:")
    )
    if physx_schemas and not authored_properties:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} has PhysxJointAPI without represented authored opinions",
        )
    unrepresented = [
        name
        for name in authored_properties
        if name not in {_PHYSX_MAX_JOINT_VELOCITY, _PHYSX_JOINT_FRICTION}
    ]
    if unrepresented:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} has unrepresented PhysxJointAPI opinions: {unrepresented}",
        )
    if authored_properties and _PHYSX_JOINT_SCHEMA not in physx_schemas:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} authored PhysxJointAPI properties without PhysxJointAPI",
        )

    max_joint_velocity = None
    physx_drive_properties: tuple[str, ...] = ()
    if _PHYSX_MAX_JOINT_VELOCITY in authored_properties:
        if not has_drive:
            _fail(
                "unsupported_optional_schema",
                f"{joint_path} max joint velocity requires a drive",
            )
        attribute = prim.GetAttribute(_PHYSX_MAX_JOINT_VELOCITY)
        _is_authored_value_only_attribute(attribute, owner_path=joint_path)
        _require_static_attribute(attribute, owner_path=joint_path)
        value = _optional_number(attribute.Get())
        if value is None or not math.isfinite(value) or value < 0.0:
            _fail(
                "invalid_drive_value",
                f"{joint_path} has invalid max joint velocity {attribute.Get()!r}",
            )
        max_joint_velocity = value
        physx_drive_properties = (_PHYSX_MAX_JOINT_VELOCITY,)

    joint_friction = None
    if _PHYSX_JOINT_FRICTION in authored_properties:
        if joint_type not in {"revolute", "prismatic"}:
            _fail(
                "joint_friction_not_applicable",
                f"{joint_path} {joint_type} joint cannot represent scalar joint friction",
            )
        attribute = prim.GetAttribute(_PHYSX_JOINT_FRICTION)
        _is_authored_value_only_attribute(attribute, owner_path=joint_path)
        _require_static_attribute(attribute, owner_path=joint_path)
        value = _optional_number(attribute.Get())
        if value is None or not math.isfinite(value) or value < 0.0:
            _fail(
                "invalid_joint_friction",
                f"{joint_path} has invalid joint friction {attribute.Get()!r}",
            )
        assert value is not None
        joint_friction = JointFrictionV1(
            coefficient=value,
            provenance=FieldProvenanceV1(
                source="authored_reference",
                artifact=reference_identity,
                prim_path=joint_path,
                properties=(_PHYSX_JOINT_FRICTION,),
                evidence=f"Authored PhysX joint friction on {joint_path}.",
            ),
        )
    return max_joint_velocity, physx_drive_properties, joint_friction


def _extract_rigid_body_plans(
    stage: Any,
    *,
    source_stage: Any,
    body_paths: set[str],
    reference_identity: ArtifactIdentityV1,
    UsdPhysics: Any,
    UsdGeom: Any,
) -> tuple[RigidBodyPlanV1, ...]:
    kilograms_per_unit = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    active_prims = tuple(
        prim for prim in _traverse_all_prims(stage) if _is_active_defined_prim(prim)
    )
    all_rigid_body_paths = {
        str(prim.GetPath())
        for prim in active_prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    owned_prims_by_body = _prims_by_nearest_body_owner(
        active_prims,
        all_rigid_body_paths,
    )
    plans = []
    for body_path in sorted(body_paths):
        prim = stage.GetPrimAtPath(body_path)
        owned_prims = owned_prims_by_body.get(body_path, ())
        _reject_unrepresented_relationships(
            prim,
            owner_path=body_path,
            relationship_names=("physics:simulationOwner",),
            unsupported_code="unsupported_rigid_body_relationship",
            schema_label="RigidBodyAPI",
        )
        source_prim = source_stage.GetPrimAtPath(body_path)
        if _is_active_defined_prim(source_prim):
            _reject_unrepresented_relationships(
                source_prim,
                owner_path=body_path,
                relationship_names=("physics:simulationOwner",),
                unsupported_code="unsupported_rigid_body_relationship",
                schema_label="source RigidBodyAPI",
            )
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_evidence = _authored_rigid_body_properties(prim)
            if rigid_body_evidence:
                _fail(
                    "rigid_body_property_without_api",
                    f"{body_path} has RigidBodyAPI properties without applied "
                    f"RigidBodyAPI: {list(rigid_body_evidence)}",
                )
            mass_evidence = _mass_api_evidence(prim, UsdPhysics=UsdPhysics)
            if mass_evidence:
                _fail(
                    "mass_without_rigid_body",
                    f"{body_path} has MassAPI evidence without RigidBodyAPI: "
                    f"{list(mass_evidence)}",
                )
            descendant_mass = _descendant_mass_contributors(
                tuple(
                    candidate
                    for candidate in active_prims
                    if _is_path_under(str(candidate.GetPath()), body_path)
                ),
                source_stage=source_stage,
                body_path=body_path,
                UsdPhysics=UsdPhysics,
            )
            if descendant_mass:
                _fail(
                    "unsupported_descendant_mass",
                    f"{body_path} has no RigidBodyAPI but its subtree has MassAPI "
                    f"evidence: {_mass_contributor_detail(descendant_mass)}",
                )
            collision_evidence = _unowned_collision_evidence(
                stage,
                body_path=body_path,
                all_rigid_body_paths=all_rigid_body_paths,
                UsdPhysics=UsdPhysics,
            )
            if collision_evidence:
                _fail(
                    "collision_without_rigid_body",
                    f"{body_path} has collider evidence without RigidBodyAPI: "
                    f"{list(collision_evidence)}",
                )
            continue
        _require_unrepresented_property_fallbacks(
            prim,
            owner_path=body_path,
            properties=_RIGID_BODY_FALLBACK_PROPERTIES,
            unsupported_code="unsupported_rigid_body_property",
            schema_label="RigidBodyAPI",
        )
        provenance = _provenance(
            reference_identity,
            body_path,
            ("PhysicsRigidBodyAPI",),
            f"Authored rigid-body API on {body_path}.",
        )
        plans.append(
            RigidBodyPlanV1(
                prim_path=body_path,
                mass=_extract_body_mass(
                    prim,
                    owned_prims=owned_prims,
                    source_stage=source_stage,
                    body_path=body_path,
                    reference_identity=reference_identity,
                    kilograms_per_unit=kilograms_per_unit,
                    meters_per_unit=meters_per_unit,
                    UsdPhysics=UsdPhysics,
                ),
                colliders=_colliders_for_body(
                    owned_prims=owned_prims,
                    source_stage=source_stage,
                    reference_stage=stage,
                    body_path=body_path,
                    reference_identity=reference_identity,
                    UsdPhysics=UsdPhysics,
                    UsdGeom=UsdGeom,
                ),
                provenance=provenance,
            )
        )
    return tuple(plans)


def _require_source_joints_replay_safe(
    source_stage: Any,
    *,
    reference_stage: Any,
    UsdPhysics: Any,
) -> None:
    """Reject source joints that replay would retain but the plan cannot author.

    Selected reference joints must be absent from the paired source.  Explicitly
    omitted joints are different: replay safely retains one only when the same
    composed joint facts already exist in the reference.  Joint prims commonly
    live in a sibling ``Joints`` scope, so endpoint-subtree scans alone cannot
    enforce this invariant.
    """

    for source_prim in _traverse_all_prims(source_stage):
        if not _is_active_defined_prim(source_prim) or not source_prim.IsA(
            UsdPhysics.Joint
        ):
            continue
        targets = _joint_body_targets(source_prim)
        path = str(source_prim.GetPath())
        reference_prim = reference_stage.GetPrimAtPath(path)
        if not _is_active_defined_prim(reference_prim) or not reference_prim.IsA(
            UsdPhysics.Joint
        ):
            _fail(
                "source_joint_not_in_reference",
                f"{path} is a source joint with body targets {sorted(targets)} "
                "but is absent from the rigged reference; replay would retain "
                "an unrepresented joint",
            )
        mismatch = _composed_joint_mismatch(source_prim, reference_prim)
        if mismatch is not None:
            _fail(
                "source_joint_differs_from_reference",
                f"{path} is a source joint that differs from the explicitly "
                f"omitted reference joint: {mismatch}",
            )


def _joint_body_targets(prim: Any) -> set[str]:
    """Return composed body relationship targets for one joint prim."""

    return {
        str(target)
        for name in ("physics:body0", "physics:body1")
        for target in prim.GetRelationship(name).GetTargets()
    }


def _composed_joint_mismatch(source_prim: Any, reference_prim: Any) -> str | None:
    """Describe the first composed joint fact that replay would not reproduce."""

    source_type = str(source_prim.GetTypeName())
    reference_type = str(reference_prim.GetTypeName())
    if source_type != reference_type:
        return f"typeName={source_type!r}/{reference_type!r}"

    source_schemas = _applied_schema_tokens(source_prim)
    reference_schemas = _applied_schema_tokens(reference_prim)
    if source_schemas != reference_schemas:
        return (
            f"appliedSchemas={sorted(source_schemas)!r}/{sorted(reference_schemas)!r}"
        )

    source_properties = _authored_property_names(source_prim)
    reference_properties = _authored_property_names(reference_prim)
    if source_properties != reference_properties:
        return (
            f"authoredProperties={sorted(source_properties)!r}/"
            f"{sorted(reference_properties)!r}"
        )

    for name in sorted(source_properties):
        source_attribute = source_prim.GetAttribute(name)
        reference_attribute = reference_prim.GetAttribute(name)
        if source_attribute and source_attribute.IsValid():
            if not reference_attribute or not reference_attribute.IsValid():
                return f"{name} changes property kind"
            if not _matching_authored_attribute(
                source_attribute,
                reference_attribute,
            ):
                return f"attribute {name} differs"
            continue

        source_relationship = source_prim.GetRelationship(name)
        reference_relationship = reference_prim.GetRelationship(name)
        if (
            not source_relationship
            or not source_relationship.IsValid()
            or not reference_relationship
            or not reference_relationship.IsValid()
            or bool(source_relationship.IsCustom())
            != bool(reference_relationship.IsCustom())
            or tuple(source_relationship.GetTargets())
            != tuple(reference_relationship.GetTargets())
        ):
            return f"relationship {name} differs"
    return None


def _require_source_physics_subset(
    source_stage: Any,
    *,
    reference_stage: Any,
    body_paths: set[str],
    UsdPhysics: Any,
    UsdGeom: Any,
) -> None:
    """Require every relevant source physics fact to survive plan replay.

    The paired source may omit facts that the rigged reference adds. Facts that
    already exist in the source are different: authoring the reference-derived
    plan does not remove them, so an absent or contradictory reference fact
    would make replay diverge from the oracle.
    """

    api_schemas = (
        ("PhysicsRigidBodyAPI", UsdPhysics.RigidBodyAPI),
        ("PhysicsMassAPI", UsdPhysics.MassAPI),
        ("PhysicsCollisionAPI", UsdPhysics.CollisionAPI),
        ("PhysicsMeshCollisionAPI", UsdPhysics.MeshCollisionAPI),
    )
    mismatches = []
    for source_prim in _traverse_all_prims(source_stage):
        if not _is_active_defined_prim(source_prim):
            continue
        path = str(source_prim.GetPath())
        if not any(
            _is_path_under(path, body_path) or _is_path_under(body_path, path)
            for body_path in body_paths
        ):
            continue

        authored_names = {
            str(prop.GetName()) for prop in source_prim.GetAuthoredProperties()
        }
        if "physics:simulationOwner" in authored_names:
            if path in body_paths:
                _fail(
                    "unsupported_rigid_body_relationship",
                    f"{path} has unrepresented source RigidBodyAPI relationships: "
                    "['physics:simulationOwner']",
                )
            _fail(
                "unsupported_collision_relationship",
                f"{path} has unrepresented source CollisionAPI relationships: "
                "['physics:simulationOwner']",
            )

        reference_prim = reference_stage.GetPrimAtPath(path)
        reference_exists = _is_active_defined_prim(reference_prim)
        for token, schema in api_schemas:
            if not _prim_has_api_fact(source_prim, token=token, schema=schema):
                continue
            if not reference_exists or not _prim_has_api_fact(
                reference_prim,
                token=token,
                schema=schema,
            ):
                mismatches.append(f"{path} {token}")

        for name in sorted(authored_names & _RIGGER_OWNED_PHYSICS_PROPERTIES):
            if not reference_exists or not _matching_authored_attribute(
                source_prim.GetAttribute(name),
                reference_prim.GetAttribute(name),
            ):
                mismatches.append(f"{path} {name}")

    # Ancestor physics cannot be authored by an endpoint-local plan. Unlike
    # planned bodies, every owned fact there must therefore already exist
    # exactly in the paired source rather than being a reference-only addition.
    for reference_prim in _traverse_all_prims(reference_stage):
        if not _is_active_defined_prim(reference_prim):
            continue
        path = str(reference_prim.GetPath())
        if path in body_paths or not any(
            _is_path_under(body_path, path) for body_path in body_paths
        ):
            continue
        authored_names = {
            str(prop.GetName()) for prop in reference_prim.GetAuthoredProperties()
        }
        if "physics:simulationOwner" in authored_names:
            _fail(
                "unsupported_rigid_body_relationship",
                f"{path} has unrepresented ancestor physics relationships: "
                "['physics:simulationOwner']",
            )
        source_prim = source_stage.GetPrimAtPath(path)
        source_exists = _is_active_defined_prim(source_prim)
        for token, schema in api_schemas:
            if not _prim_has_api_fact(reference_prim, token=token, schema=schema):
                continue
            if not source_exists or not _prim_has_api_fact(
                source_prim,
                token=token,
                schema=schema,
            ):
                mismatches.append(f"{path} {token}")
        for name in sorted(authored_names & _RIGGER_OWNED_PHYSICS_PROPERTIES):
            if not source_exists or not _matching_authored_attribute(
                source_prim.GetAttribute(name),
                reference_prim.GetAttribute(name),
            ):
                mismatches.append(f"{path} {name}")

    articulation_token = "PhysicsArticulationRootAPI"
    for source_prim in _traverse_all_prims(source_stage):
        if not _is_active_defined_prim(source_prim) or not _prim_has_api_fact(
            source_prim,
            token=articulation_token,
            schema=UsdPhysics.ArticulationRootAPI,
        ):
            continue
        path = str(source_prim.GetPath())
        if not any(_is_path_under(body_path, path) for body_path in body_paths):
            continue
        reference_prim = reference_stage.GetPrimAtPath(path)
        if not _is_active_defined_prim(reference_prim) or not _prim_has_api_fact(
            reference_prim,
            token=articulation_token,
            schema=UsdPhysics.ArticulationRootAPI,
        ):
            mismatches.append(f"{path} {articulation_token}")

    if mismatches:
        _fail(
            "source_physics_not_in_reference",
            "paired source physics facts are absent or differ in the rigged "
            f"reference: {sorted(set(mismatches))}",
        )
    _require_unplanned_nested_body_facts_preexisting(
        source_stage,
        reference_stage=reference_stage,
        body_paths=body_paths,
        UsdPhysics=UsdPhysics,
        UsdGeom=UsdGeom,
        api_schemas=api_schemas,
    )


def _require_unplanned_nested_body_facts_preexisting(
    source_stage: Any,
    *,
    reference_stage: Any,
    body_paths: set[str],
    UsdPhysics: Any,
    UsdGeom: Any,
    api_schemas: tuple[tuple[str, Any], ...],
) -> None:
    """Reject unplanned nested-body subtree facts that replay cannot author."""

    reference_prims = {
        str(prim.GetPath()): prim
        for prim in _traverse_all_prims(reference_stage)
        if _is_active_defined_prim(prim)
    }
    source_prims = {
        str(prim.GetPath()): prim
        for prim in _traverse_all_prims(source_stage)
        if _is_active_defined_prim(prim)
    }
    reference_rigid_bodies = {
        path
        for path, prim in reference_prims.items()
        if _prim_has_api_fact(
            prim,
            token="PhysicsRigidBodyAPI",
            schema=UsdPhysics.RigidBodyAPI,
        )
    }
    nested_bodies = sorted(
        path
        for path in reference_rigid_bodies - body_paths
        if any(_is_path_under(path, body_path) for body_path in body_paths)
    )
    subtree_api_schemas = (
        *api_schemas,
        ("PhysicsArticulationRootAPI", UsdPhysics.ArticulationRootAPI),
    )
    candidate_paths = set(source_prims) | set(reference_prims)
    for body_path in nested_bodies:
        owned_paths = sorted(
            path
            for path in candidate_paths
            if _nearest_body_owner(path, reference_rigid_bodies) == body_path
        )
        for path in owned_paths:
            source_prim = source_prims.get(path)
            reference_prim = reference_prims.get(path)
            source_apis = _physics_api_facts(source_prim, subtree_api_schemas)
            reference_apis = _physics_api_facts(
                reference_prim,
                subtree_api_schemas,
            )
            source_authored = _authored_property_names(source_prim)
            reference_authored = _authored_property_names(reference_prim)
            if "physics:simulationOwner" in source_authored | reference_authored:
                _fail(
                    "unsupported_rigid_body_relationship",
                    f"{path} has unrepresented nested RigidBodyAPI relationships: "
                    "['physics:simulationOwner']",
                )
            source_properties = {
                name for name in source_authored if _is_physics_property_name(name)
            }
            reference_properties = {
                name for name in reference_authored if _is_physics_property_name(name)
            }
            attributes_match = (
                source_properties == reference_properties
                and (not source_properties or source_prim is not None)
                and (not reference_properties or reference_prim is not None)
                and all(
                    _matching_authored_property(
                        source_prim,
                        reference_prim,
                        name=name,
                    )
                    for name in source_properties
                    if source_prim is not None and reference_prim is not None
                )
            )
            if source_apis != reference_apis or not attributes_match:
                _fail(
                    "unrepresented_descendant_rigid_body",
                    "reference nested rigid-body subtree facts are not fully "
                    f"preexisting in the paired source under {body_path} at {path}: "
                    f"source_apis={list(source_apis)}, "
                    f"reference_apis={list(reference_apis)}, "
                    f"source_properties={sorted(source_properties)}, "
                    f"reference_properties={sorted(reference_properties)}",
                )
            if "PhysicsCollisionAPI" in reference_apis:
                assert source_prim is not None and reference_prim is not None
                _require_compatible_source_collider(
                    reference_prim,
                    source_stage=source_stage,
                    reference_stage=reference_stage,
                    body_path=body_path,
                    path=path,
                    UsdGeom=UsdGeom,
                )


def _physics_api_facts(
    prim: Any | None,
    api_schemas: tuple[tuple[str, Any], ...],
) -> tuple[str, ...]:
    if prim is None:
        return ()
    facts = [
        token for token in _applied_schema_tokens(prim) if _is_physics_api_token(token)
    ]
    for token, schema in api_schemas:
        if token not in facts and _prim_has_api_fact(prim, token=token, schema=schema):
            facts.append(token)
    return tuple(facts)


def _require_unmodeled_physics_facts_preexisting(
    source_stage: Any,
    *,
    reference_stage: Any,
    body_paths: set[str],
    rigid_bodies: tuple[RigidBodyPlanV1, ...],
    articulation_root: ArticulationRootPlanV1 | None,
    UsdPhysics: Any,
) -> None:
    """Require non-v1 physics facts in replay scope to preexist exactly."""

    modeled_apis: dict[str, list[str]] = {}
    modeled_properties: dict[str, set[str]] = {}
    lifted_mass_contributor_paths: set[str] = set()

    def add_modeled_api(path: str, token: str) -> None:
        tokens = modeled_apis.setdefault(path, [])
        if token not in tokens:
            tokens.append(token)

    def add_modeled_properties(path: str, names: Iterable[str]) -> None:
        modeled_properties.setdefault(path, set()).update(names)

    for body in rigid_bodies:
        add_modeled_api(body.prim_path, "PhysicsRigidBodyAPI")
        add_modeled_properties(body.prim_path, _RIGID_BODY_FALLBACK_PROPERTIES)
        if body.mass is not None:
            mass_properties = {"physics:mass", "physics:diagonalInertia"}
            if body.mass.center_of_mass_m is not None:
                mass_properties.add("physics:centerOfMass")
            if body.mass.principal_axes is not None:
                mass_properties.add("physics:principalAxes")
            contributor_path = body.mass.provenance.prim_path
            if contributor_path == body.prim_path:
                add_modeled_api(body.prim_path, "PhysicsMassAPI")
                add_modeled_properties(body.prim_path, mass_properties)
            elif contributor_path is not None:
                add_modeled_api(contributor_path, "PhysicsMassAPI")
                lifted_mass_contributor_paths.add(contributor_path)
                add_modeled_properties(
                    contributor_path,
                    body.mass.provenance.properties,
                )
        for collider in body.colliders:
            add_modeled_api(collider.prim_path, "PhysicsCollisionAPI")
            add_modeled_properties(
                collider.prim_path,
                _COLLISION_FALLBACK_PROPERTIES,
            )
            if collider.has_mesh_collision_api:
                add_modeled_api(collider.prim_path, "PhysicsMeshCollisionAPI")
            if collider.mesh_approximation is not None:
                add_modeled_properties(
                    collider.prim_path,
                    ("physics:approximation",),
                )
    if articulation_root is not None:
        add_modeled_api(
            articulation_root.prim_path,
            "PhysicsArticulationRootAPI",
        )
    # Contributor APIs are evidence at a retained source location, not APIs the
    # owner authorer appends. Preserve their exact reference order for the
    # source-prefix replay check below; owner APIs keep canonical author order.
    for contributor_path in lifted_mass_contributor_paths:
        reference_prim = reference_stage.GetPrimAtPath(contributor_path)
        if not reference_prim or not reference_prim.IsValid():
            _fail(
                "lifted_mass_contributor_missing",
                f"planned lifted mass contributor {contributor_path} does not "
                "resolve to a valid reference prim",
            )
        reference_order = {
            token: index
            for index, token in enumerate(_applied_schema_tokens(reference_prim))
        }
        modeled_apis[contributor_path].sort(
            key=lambda token: reference_order.get(token, len(reference_order))
        )

    source_prims = {
        str(prim.GetPath()): prim
        for prim in _traverse_all_prims(source_stage)
        if _is_active_defined_prim(prim)
    }
    reference_prims = {
        str(prim.GetPath()): prim
        for prim in _traverse_all_prims(reference_stage)
        if _is_active_defined_prim(prim)
    }
    replay_scope_paths = {
        path
        for path in set(source_prims) | set(reference_prims)
        if any(
            _is_path_under(path, body_path) or _is_path_under(body_path, path)
            for body_path in body_paths
        )
    }
    material_paths = _require_physics_material_bindings_preexisting(
        source_stage,
        reference_stage=reference_stage,
        source_prims=source_prims,
        reference_prims=reference_prims,
        replay_scope_paths=replay_scope_paths,
    )
    for path in sorted(set(source_prims) | set(reference_prims)):
        source_prim = source_prims.get(path)
        reference_prim = reference_prims.get(path)
        if path not in replay_scope_paths and path not in material_paths:
            continue

        source_apis = _physics_api_facts(source_prim, ())
        reference_apis = _physics_api_facts(reference_prim, ())
        modeled_api_tokens = modeled_apis.get(path, [])
        source_api_tokens = set(source_apis)
        source_prefix_matches = reference_apis[: len(source_apis)] == source_apis
        reference_append_apis = reference_apis[len(source_apis) :]
        expected_reference_append_apis = tuple(
            token for token in modeled_api_tokens if token not in source_api_tokens
        )
        append_only_modeled = reference_append_apis == expected_reference_append_apis
        source_properties = {
            name
            for name in _authored_property_names(source_prim)
            if _is_physics_property_name(name)
        } - modeled_properties.get(path, set())
        reference_properties = {
            name
            for name in _authored_property_names(reference_prim)
            if _is_physics_property_name(name)
        } - modeled_properties.get(path, set())
        mismatched_properties = sorted(
            name
            for name in source_properties & reference_properties
            if source_prim is None
            or reference_prim is None
            or not _matching_authored_property(
                source_prim,
                reference_prim,
                name=name,
            )
        )
        if (
            not source_prefix_matches
            or not append_only_modeled
            or source_properties != reference_properties
            or mismatched_properties
        ):
            _fail(
                "unrepresented_physics_facts",
                f"{path} has non-v1 physics facts that are not exact "
                "preexisting source/reference parity: "
                f"source_apis={list(source_apis)}, "
                f"reference_apis={list(reference_apis)}, "
                f"reference_append_apis={list(reference_append_apis)}, "
                "expected_reference_append_apis="
                f"{list(expected_reference_append_apis)}, "
                f"source_prefix_matches={source_prefix_matches}, "
                f"append_only_modeled={append_only_modeled}, "
                f"source_properties={sorted(source_properties)}, "
                f"reference_properties={sorted(reference_properties)}, "
                f"mismatched_properties={mismatched_properties}",
            )


def _require_physics_material_bindings_preexisting(
    source_stage: Any,
    *,
    reference_stage: Any,
    source_prims: dict[str, Any],
    reference_prims: dict[str, Any],
    replay_scope_paths: set[str],
) -> set[str]:
    """Require effective physics-material binding inputs to preexist exactly.

    Physics-purpose bindings are not represented by the v1 authoring plan. A
    binding on a replay-scope prim and the external material or collection it
    references must therefore be identical in the source and reference.
    All-purpose bindings are included when either bound target carries physics
    facts because USD material-purpose resolution can fall back to them.
    """

    material_paths: set[str] = set()
    checked_collections: set[tuple[str, str]] = set()
    for owner_path in sorted(replay_scope_paths):
        source_prim = source_prims.get(owner_path)
        reference_prim = reference_prims.get(owner_path)
        relevant_binding_names: set[str] = set()
        names = {
            name
            for prim in (source_prim, reference_prim)
            for name in _authored_property_names(prim)
            if name.startswith("material:binding")
        }
        for name in sorted(names):
            explicit_physics = _is_explicit_physics_material_binding_name(name)
            all_purpose = _is_all_purpose_material_binding_name(name)
            if not explicit_physics and not all_purpose:
                continue
            source_target = _material_binding_target(
                source_prim,
                relationship_name=name,
                strict=explicit_physics,
            )
            reference_target = _material_binding_target(
                reference_prim,
                relationship_name=name,
                strict=explicit_physics,
            )
            if (
                all_purpose
                and not explicit_physics
                and not any(
                    _material_binding_may_target_physics(
                        prim,
                        source_stage,
                        reference_stage=reference_stage,
                        relationship_name=name,
                    )
                    for prim in (source_prim, reference_prim)
                )
            ):
                continue
            if all_purpose and not explicit_physics:
                source_target = _material_binding_target(
                    source_prim,
                    relationship_name=name,
                    strict=True,
                )
                reference_target = _material_binding_target(
                    reference_prim,
                    relationship_name=name,
                    strict=True,
                )
            if source_target is None or reference_target is None:
                _fail(
                    "unrepresented_physics_material_binding",
                    f"{owner_path} {name} does not preexist in both paired stages",
                )
            assert source_target is not None and reference_target is not None
            if source_target != reference_target or not _matching_authored_property(
                source_prim,
                reference_prim,
                name=name,
            ):
                _fail(
                    "unrepresented_physics_material_binding",
                    f"{owner_path} {name} differs between paired stages: "
                    f"source={source_target}, reference={reference_target}",
                )
            _require_material_binding_api_preexisting(
                source_prim,
                reference_prim,
                owner_path=owner_path,
                relationship_name=name,
            )
            _require_material_target_resolved(
                source_stage,
                reference_stage=reference_stage,
                owner_path=owner_path,
                relationship_name=name,
                material_path=source_target.material_path,
            )
            relevant_binding_names.add(name)
            material_paths.add(source_target.material_path)
            if source_target.collection_prim_path is not None:
                assert source_target.collection_instance is not None
                collection_key = (
                    source_target.collection_prim_path,
                    source_target.collection_instance,
                )
                if collection_key not in checked_collections:
                    _require_collection_definition_preexisting(
                        source_stage,
                        reference_stage=reference_stage,
                        owner_path=owner_path,
                        relationship_name=name,
                        collection_prim_path=source_target.collection_prim_path,
                        collection_instance=source_target.collection_instance,
                        checked_collections=checked_collections,
                    )
        if relevant_binding_names:
            assert source_prim is not None and reference_prim is not None
            _require_material_binding_order_preexisting(
                source_prim,
                reference_prim,
                owner_path=owner_path,
                binding_names=relevant_binding_names,
            )
    return material_paths


def _require_material_binding_order_preexisting(
    source_prim: Any,
    reference_prim: Any,
    *,
    owner_path: str,
    binding_names: set[str],
) -> None:
    """Require effective precedence among replay-relevant bindings to match."""

    source_order = tuple(
        str(prop.GetName())
        for prop in source_prim.GetProperties()
        if str(prop.GetName()) in binding_names
    )
    reference_order = tuple(
        str(prop.GetName())
        for prop in reference_prim.GetProperties()
        if str(prop.GetName()) in binding_names
    )
    if source_order != reference_order:
        _fail(
            "unrepresented_physics_material_binding",
            f"{owner_path} material binding precedence differs between paired "
            f"stages: source={source_order}, reference={reference_order}",
        )


def _is_explicit_physics_material_binding_name(name: str) -> bool:
    """Return whether a property claims a physics-purpose material binding."""

    if name == _DIRECT_PHYSICS_MATERIAL_BINDING:
        return True
    if name.startswith(_COLLECTION_PHYSICS_MATERIAL_BINDING_PREFIX):
        return True
    return name.startswith("material:binding:") and "physics" in name.split(":")[2:]


def _is_all_purpose_material_binding_name(name: str) -> bool:
    """Return whether a property has one supported all-purpose binding shape."""

    if name == "material:binding":
        return True
    parts = name.split(":")
    return (
        len(parts) == 4
        and parts[:3] == ["material", "binding", "collection"]
        and bool(parts[3])
    )


def _material_binding_target(
    prim: Any | None,
    *,
    relationship_name: str,
    strict: bool,
) -> _PhysicsMaterialBindingTarget | None:
    """Parse one supported direct or collection material binding."""

    if prim is None:
        return None
    if relationship_name not in _authored_property_names(prim):
        return None
    relationship = prim.GetRelationship(relationship_name)
    if not relationship or not relationship.IsValid():
        if strict:
            _fail(
                "unsupported_physics_material_binding",
                f"{prim.GetPath()} {relationship_name} is not a relationship",
            )
        return None
    targets = tuple(relationship.GetTargets())
    parts = relationship_name.split(":")
    is_direct = relationship_name in {
        "material:binding",
        _DIRECT_PHYSICS_MATERIAL_BINDING,
    }
    is_collection = (
        len(parts) == 4
        and parts[:3] == ["material", "binding", "collection"]
        and bool(parts[3])
    ) or (
        len(parts) == 5
        and parts[:4] == ["material", "binding", "collection", "physics"]
        and bool(parts[4])
    )
    expected_targets = 1 if is_direct else 2 if is_collection else 0
    if not expected_targets or len(targets) != expected_targets:
        if strict:
            _fail(
                "unsupported_physics_material_binding",
                f"{prim.GetPath()} {relationship_name} has unsupported shape or "
                f"targets: {[str(target) for target in targets]}",
            )
        return None
    material_target = targets[-1]
    if (
        not material_target.IsAbsolutePath()
        or not material_target.IsPrimPath()
        or material_target.IsAbsoluteRootPath()
    ):
        if strict:
            _fail(
                "unsupported_physics_material_binding",
                f"{prim.GetPath()} {relationship_name} has invalid material "
                f"target: {material_target}",
            )
        return None
    if is_direct:
        return _PhysicsMaterialBindingTarget(material_path=str(material_target))

    collection_target = targets[0]
    collection_property = str(collection_target.name)
    if (
        not collection_target.IsAbsolutePath()
        or not collection_target.IsPropertyPath()
        or collection_target.IsAbsoluteRootPath()
        or not collection_property.startswith("collection:")
        or not collection_property.removeprefix("collection:")
    ):
        if strict:
            _fail(
                "unsupported_physics_material_binding",
                f"{prim.GetPath()} {relationship_name} has invalid collection "
                f"target: {collection_target}",
            )
        return None
    return _PhysicsMaterialBindingTarget(
        material_path=str(material_target),
        collection_prim_path=str(collection_target.GetPrimPath()),
        collection_instance=collection_property.removeprefix("collection:"),
    )


def _material_binding_may_target_physics(
    prim: Any | None,
    source_stage: Any,
    *,
    reference_stage: Any,
    relationship_name: str,
) -> bool:
    """Return whether a binding has any target carrying physics facts."""

    if prim is None:
        return False
    relationship = prim.GetRelationship(relationship_name)
    if not relationship or not relationship.IsValid():
        return False
    material_paths = {
        str(target)
        for target in relationship.GetTargets()
        if target.IsAbsolutePath()
        and target.IsPrimPath()
        and not target.IsAbsoluteRootPath()
    }
    return any(
        _material_target_has_physics_facts(
            source_stage,
            reference_stage=reference_stage,
            material_path=material_path,
        )
        for material_path in material_paths
    )


def _material_target_has_physics_facts(
    source_stage: Any,
    *,
    reference_stage: Any,
    material_path: str,
) -> bool:
    """Return whether either paired material target carries physics facts."""

    for stage in (source_stage, reference_stage):
        prim = stage.GetPrimAtPath(material_path)
        if not _is_active_defined_prim(prim):
            continue
        if _physics_api_facts(prim, ()) or any(
            _is_physics_property_name(name) for name in _authored_property_names(prim)
        ):
            return True
    return False


def _require_material_binding_api_preexisting(
    source_prim: Any,
    reference_prim: Any,
    *,
    owner_path: str,
    relationship_name: str,
) -> None:
    """Require a replay-relevant relationship to use MaterialBindingAPI."""

    token = "MaterialBindingAPI"
    source_has_api = token in _applied_schema_tokens(source_prim)
    reference_has_api = token in _applied_schema_tokens(reference_prim)
    if not source_has_api or not reference_has_api:
        _fail(
            "unsupported_physics_material_binding",
            f"{owner_path} {relationship_name} requires exact {token} presence "
            f"in both paired stages: source={source_has_api}, "
            f"reference={reference_has_api}",
        )


def _require_material_target_resolved(
    source_stage: Any,
    *,
    reference_stage: Any,
    owner_path: str,
    relationship_name: str,
    material_path: str,
) -> None:
    """Require a replay-relevant target to be a material in both stages."""

    from pxr import UsdShade

    missing = [
        label
        for label, stage in (("source", source_stage), ("reference", reference_stage))
        if not _is_active_defined_prim(stage.GetPrimAtPath(material_path))
    ]
    if missing:
        _fail(
            "unrepresented_physics_material_binding",
            f"{owner_path} {relationship_name} material target {material_path} "
            f"does not resolve as an active defined prim in {missing}",
        )
    invalid_types = [
        (label, str(prim.GetTypeName()))
        for label, stage in (("source", source_stage), ("reference", reference_stage))
        if not (prim := stage.GetPrimAtPath(material_path)).IsA(UsdShade.Material)
    ]
    if invalid_types:
        _fail(
            "unsupported_physics_material_binding",
            f"{owner_path} {relationship_name} target {material_path} must be a "
            f"UsdShade.Material in both paired stages: {invalid_types}",
        )


def _require_collection_definition_preexisting(
    source_stage: Any,
    *,
    reference_stage: Any,
    owner_path: str,
    relationship_name: str,
    collection_prim_path: str,
    collection_instance: str,
    checked_collections: set[tuple[str, str]],
) -> None:
    """Require a physics binding's bounded collection closure to match."""

    root_key = (collection_prim_path, collection_instance)
    pending = [(root_key, 0)]
    definitions_checked = 0

    while pending:
        collection_key, depth = pending.pop()
        if collection_key in checked_collections:
            continue
        if depth > _MAX_PHYSICS_MATERIAL_COLLECTION_DEPTH:
            _fail(
                "unsupported_physics_material_binding",
                f"{owner_path} {relationship_name} collection closure exceeds "
                "the maximum nesting depth of "
                f"{_MAX_PHYSICS_MATERIAL_COLLECTION_DEPTH}: {collection_key}",
            )
        if definitions_checked >= _MAX_PHYSICS_MATERIAL_COLLECTION_DEFINITIONS:
            _fail(
                "unsupported_physics_material_binding",
                f"{owner_path} {relationship_name} collection closure exceeds "
                "the maximum definition count of "
                f"{_MAX_PHYSICS_MATERIAL_COLLECTION_DEFINITIONS}",
            )
        checked_collections.add(collection_key)
        definitions_checked += 1
        direct_nested_keys, indirect_nested_keys = (
            _matching_collection_definition_nested_keys(
                source_stage,
                reference_stage=reference_stage,
                owner_path=owner_path,
                relationship_name=relationship_name,
                collection_prim_path=collection_key[0],
                collection_instance=collection_key[1],
            )
        )
        # Indirect query dependencies remain fail-closed coverage for membership
        # expressions. Push them first so direct edges are visited first and
        # retain their true nesting depth.
        pending.extend(
            (nested_key, depth + 1)
            for nested_key in reversed(indirect_nested_keys)
            if nested_key not in checked_collections
        )
        pending.extend(
            (nested_key, depth + 1)
            for nested_key in reversed(direct_nested_keys)
            if nested_key not in checked_collections
        )


def _matching_collection_definition_nested_keys(
    source_stage: Any,
    *,
    reference_stage: Any,
    owner_path: str,
    relationship_name: str,
    collection_prim_path: str,
    collection_instance: str,
) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    """Validate one definition and return direct and indirect nested keys."""

    from pxr import Sdf, Usd

    source_prim = source_stage.GetPrimAtPath(collection_prim_path)
    reference_prim = reference_stage.GetPrimAtPath(collection_prim_path)
    if not _is_active_defined_prim(source_prim) or not _is_active_defined_prim(
        reference_prim
    ):
        _fail(
            "unsupported_physics_material_binding",
            f"{owner_path} {relationship_name} collection target does not resolve "
            f"in both paired stages: {collection_prim_path}",
        )
    api_token = f"CollectionAPI:{collection_instance}"
    if api_token not in _applied_schema_tokens(
        source_prim
    ) or api_token not in _applied_schema_tokens(reference_prim):
        _fail(
            "unsupported_physics_material_binding",
            f"{owner_path} {relationship_name} requires {api_token} on "
            f"{collection_prim_path} in both paired stages",
        )
    property_prefix = f"collection:{collection_instance}:"
    source_properties = {
        name
        for name in _authored_property_names(source_prim)
        if name.startswith(property_prefix)
    }
    reference_properties = {
        name
        for name in _authored_property_names(reference_prim)
        if name.startswith(property_prefix)
    }
    mismatched = sorted(
        name
        for name in source_properties & reference_properties
        if not _matching_authored_property(
            source_prim,
            reference_prim,
            name=name,
        )
    )
    if source_properties != reference_properties or mismatched:
        _fail(
            "unrepresented_physics_material_binding",
            f"{owner_path} {relationship_name} collection definition differs at "
            f"{collection_prim_path}: source={sorted(source_properties)}, "
            f"reference={sorted(reference_properties)}, mismatched={mismatched}",
        )

    collection_path = Sdf.Path(collection_prim_path).AppendProperty(
        f"collection:{collection_instance}"
    )
    source_collection = Usd.CollectionAPI.Get(source_stage, collection_path)
    reference_collection = Usd.CollectionAPI.Get(reference_stage, collection_path)
    source_valid, source_reason = source_collection.Validate()
    reference_valid, reference_reason = reference_collection.Validate()
    if not source_valid or not reference_valid:
        _fail(
            "unsupported_physics_material_binding",
            f"{owner_path} {relationship_name} collection {collection_path} is "
            "invalid in a paired stage: "
            f"source={source_reason!r}, reference={reference_reason!r}",
        )
    try:
        source_query = source_collection.ComputeMembershipQuery()
        reference_query = reference_collection.ComputeMembershipQuery()
        source_membership = tuple(
            sorted(
                (str(path), str(rule))
                for path, rule in source_query.GetAsPathExpansionRuleMap().items()
            )
        )
        reference_membership = tuple(
            sorted(
                (str(path), str(rule))
                for path, rule in reference_query.GetAsPathExpansionRuleMap().items()
            )
        )
        source_included_collections = tuple(
            sorted(str(path) for path in source_query.GetIncludedCollections())
        )
        reference_included_collections = tuple(
            sorted(str(path) for path in reference_query.GetIncludedCollections())
        )
    except Exception as exc:
        raise JointRiggerContractError(
            "unsupported_physics_material_binding",
            f"could not resolve collection membership for {collection_path}: {exc}",
        ) from exc
    if (
        source_membership != reference_membership
        or source_included_collections != reference_included_collections
    ):
        _fail(
            "unrepresented_physics_material_binding",
            f"{owner_path} {relationship_name} effective collection membership "
            f"differs for {collection_path}: source={source_membership}, "
            f"reference={reference_membership}, "
            f"source_collections={source_included_collections}, "
            f"reference_collections={reference_included_collections}",
        )

    included_collection_paths = []
    for nested_path_value in source_included_collections:
        nested_path = Sdf.Path(nested_path_value)
        # ComputeMembershipQuery guarantees these are CollectionAPI paths. Keep
        # the fail-closed guard for future resolver implementations.
        if not Usd.CollectionAPI.IsCollectionAPIPath(nested_path):  # pragma: no cover
            _fail(
                "unsupported_physics_material_binding",
                f"{collection_path} resolves a malformed nested collection path: "
                f"{nested_path}",
            )
        included_collection_paths.append(nested_path)

    included_collection_path_set = set(included_collection_paths)
    direct_collection_paths = {
        target_path
        for target_path in source_collection.GetIncludesRel().GetTargets()
        if target_path in included_collection_path_set
    }
    direct_keys: list[tuple[str, str]] = []
    indirect_keys: list[tuple[str, str]] = []
    for nested_path in included_collection_paths:
        nested_instance = str(nested_path.name).removeprefix("collection:")
        nested_key = (str(nested_path.GetPrimPath()), nested_instance)
        target_keys = (
            direct_keys if nested_path in direct_collection_paths else indirect_keys
        )
        target_keys.append(nested_key)
    return tuple(direct_keys), tuple(indirect_keys)


def _is_physics_api_token(token: str) -> bool:
    return token.startswith(("Physics", "Physx"))


def _is_physics_property_name(name: str) -> bool:
    return name.startswith(
        (
            "physics:",
            _DRIVE_PROPERTY_PREFIX,
            *_UNREPRESENTED_JOINT_PROPERTY_PREFIXES,
        )
    ) or name.lower().startswith("physx")


def _matching_authored_property(
    source_prim: Any,
    reference_prim: Any,
    *,
    name: str,
) -> bool:
    """Compare one authored physics attribute or relationship exactly."""

    source_attribute = source_prim.GetAttribute(name)
    reference_attribute = reference_prim.GetAttribute(name)
    if bool(source_attribute and source_attribute.IsValid()) or bool(
        reference_attribute and reference_attribute.IsValid()
    ):
        if (
            not source_attribute
            or not source_attribute.IsValid()
            or not reference_attribute
            or not reference_attribute.IsValid()
            or str(source_attribute.GetTypeName())
            != str(reference_attribute.GetTypeName())
            or bool(source_attribute.IsCustom()) != bool(reference_attribute.IsCustom())
            or str(source_attribute.GetVariability())
            != str(reference_attribute.GetVariability())
            or bool(source_attribute.HasAuthoredValueOpinion())
            != bool(reference_attribute.HasAuthoredValueOpinion())
            or tuple(str(path) for path in source_attribute.GetConnections())
            != tuple(str(path) for path in reference_attribute.GetConnections())
            or not _usd_values_equal(
                source_attribute.GetAllAuthoredMetadata(),
                reference_attribute.GetAllAuthoredMetadata(),
            )
        ):
            return False
        source_times = tuple(
            float(value) for value in source_attribute.GetTimeSamples()
        )
        reference_times = tuple(
            float(value) for value in reference_attribute.GetTimeSamples()
        )
        return bool(
            source_times == reference_times
            and _usd_values_equal(source_attribute.Get(), reference_attribute.Get())
            and all(
                _usd_values_equal(
                    source_attribute.Get(time),
                    reference_attribute.Get(time),
                )
                for time in source_times
            )
        )

    source_relationship = source_prim.GetRelationship(name)
    reference_relationship = reference_prim.GetRelationship(name)
    return bool(
        source_relationship
        and source_relationship.IsValid()
        and reference_relationship
        and reference_relationship.IsValid()
        and bool(source_relationship.IsCustom())
        == bool(reference_relationship.IsCustom())
        and tuple(source_relationship.GetTargets())
        == tuple(reference_relationship.GetTargets())
        and _usd_values_equal(
            source_relationship.GetAllAuthoredMetadata(),
            reference_relationship.GetAllAuthoredMetadata(),
        )
    )


def _authored_property_names(prim: Any | None) -> set[str]:
    if prim is None:
        return set()
    return {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}


def _prim_has_api_fact(prim: Any, *, token: str, schema: Any) -> bool:
    """Return whether a composed prim carries one owned API-schema fact."""

    return bool(prim.HasAPI(schema) or token in _applied_schema_tokens(prim))


def _matching_authored_attribute(source: Any, reference: Any) -> bool:
    """Compare one source-authored schema attribute with reference evidence."""

    if not source or not source.IsValid() or not source.HasAuthoredValueOpinion():
        return False
    if (
        not reference
        or not reference.IsValid()
        or not reference.HasAuthoredValueOpinion()
        or str(source.GetTypeName()) != str(reference.GetTypeName())
        or bool(source.IsCustom()) != bool(reference.IsCustom())
        or str(source.GetVariability()) != str(reference.GetVariability())
        or not _usd_values_equal(
            source.GetAllAuthoredMetadata(),
            reference.GetAllAuthoredMetadata(),
        )
    ):
        return False
    source_times = tuple(float(value) for value in source.GetTimeSamples())
    reference_times = tuple(float(value) for value in reference.GetTimeSamples())
    if source_times != reference_times or not _usd_values_equal(
        source.Get(),
        reference.Get(),
    ):
        return False
    if tuple(str(path) for path in source.GetConnections()) != tuple(
        str(path) for path in reference.GetConnections()
    ):
        return False
    return all(
        _usd_values_equal(source.Get(time), reference.Get(time))
        for time in source_times
    )


def _usd_values_equal(left: Any, right: Any) -> bool:
    """Compare scalar and pxr container values without lossy string coercion."""

    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _descendant_mass_contributors(
    owned_prims: Sequence[Any],
    *,
    source_stage: Any,
    body_path: str,
    UsdPhysics: Any,
) -> tuple[_DescendantMassEvidence, ...]:
    """Inventory every nearest-owner descendant MassAPI fact deterministically."""

    contributors: list[_DescendantMassEvidence] = []
    for prim in owned_prims:
        path = str(prim.GetPath())
        if path == body_path:
            continue
        reference_evidence = _mass_api_evidence(prim, UsdPhysics=UsdPhysics)
        if not reference_evidence:
            continue
        source_prim = source_stage.GetPrimAtPath(path)
        source_evidence = (
            _mass_api_evidence(source_prim, UsdPhysics=UsdPhysics)
            if _is_active_defined_prim(source_prim)
            else ()
        )
        if _matching_preexisting_mass_facts(
            source_prim,
            prim,
            UsdPhysics=UsdPhysics,
        ):
            replay_status: _DescendantMassReplayStatus = "matching_preexisting"
        elif source_evidence:
            replay_status = "source_conflict"
        else:
            replay_status = "reference_only"
        contributors.append(
            _DescendantMassEvidence(
                prim=prim,
                reference_evidence=reference_evidence,
                source_evidence=source_evidence,
                replay_status=replay_status,
            )
        )
    return tuple(sorted(contributors, key=lambda item: str(item.prim.GetPath())))


def _mass_contributor_detail(
    contributors: Sequence[_DescendantMassEvidence],
) -> str:
    return repr(
        [
            {
                "prim_path": str(item.prim.GetPath()),
                "reference_evidence": list(item.reference_evidence),
                "source_evidence": list(item.source_evidence),
                "replay_status": item.replay_status,
            }
            for item in contributors
        ]
    )


def _matching_preexisting_mass_facts(
    source_prim: Any,
    reference_prim: Any,
    *,
    UsdPhysics: Any,
) -> bool:
    """Return whether replay already preserves every reference mass fact."""

    if not _is_active_defined_prim(source_prim):
        return False
    source_has_api = _prim_has_api_fact(
        source_prim,
        token="PhysicsMassAPI",
        schema=UsdPhysics.MassAPI,
    )
    reference_has_api = _prim_has_api_fact(
        reference_prim,
        token="PhysicsMassAPI",
        schema=UsdPhysics.MassAPI,
    )
    if source_has_api != reference_has_api:
        return False
    source_properties = set(_authored_mass_properties(source_prim))
    reference_properties = set(_authored_mass_properties(reference_prim))
    return source_properties == reference_properties and all(
        _matching_authored_attribute(
            source_prim.GetAttribute(name),
            reference_prim.GetAttribute(name),
        )
        for name in source_properties
    )


def _extract_body_mass(
    prim: Any,
    *,
    owned_prims: Sequence[Any],
    source_stage: Any,
    body_path: str,
    reference_identity: ArtifactIdentityV1,
    kilograms_per_unit: float,
    meters_per_unit: float,
    UsdPhysics: Any,
) -> MassPropertiesV1 | None:
    """Prefer owner mass facts, otherwise lift one complete collider contributor."""

    contributors = _descendant_mass_contributors(
        owned_prims,
        source_stage=source_stage,
        body_path=body_path,
        UsdPhysics=UsdPhysics,
    )
    owner_evidence = _mass_api_evidence(prim, UsdPhysics=UsdPhysics)
    if owner_evidence and contributors:
        _fail(
            "body_descendant_mass_conflict",
            f"planned rigid-body owner {body_path} has body-level mass evidence "
            f"{list(owner_evidence)} and descendant MassAPI evidence: "
            f"{_mass_contributor_detail(contributors)}",
        )
    owner_mass = _extract_mass(
        prim,
        body_path=body_path,
        reference_identity=reference_identity,
        kilograms_per_unit=kilograms_per_unit,
        meters_per_unit=meters_per_unit,
        UsdPhysics=UsdPhysics,
    )
    if owner_mass is not None:
        return owner_mass
    if not contributors:
        return None
    source_conflicts = tuple(
        item for item in contributors if item.replay_status == "source_conflict"
    )
    if source_conflicts:
        _fail(
            "descendant_mass_source_conflict",
            f"planned rigid-body owner {body_path} has descendant mass evidence "
            "that differs between source and reference; replay cannot replace or "
            f"duplicate it: {_mass_contributor_detail(contributors)}",
        )
    reference_only = tuple(
        item for item in contributors if item.replay_status == "reference_only"
    )
    replay_preserved = tuple(
        item for item in contributors if item.replay_status == "matching_preexisting"
    )
    if not reference_only:
        return None
    if len(reference_only) != 1 or replay_preserved:
        _fail(
            "multiple_descendant_mass_contributors",
            f"planned rigid-body owner {body_path} cannot choose among or combine "
            "descendant MassAPI records: "
            f"{_mass_contributor_detail(contributors)}",
        )
    contributor = reference_only[0].prim
    return _extract_lifted_descendant_mass(
        contributor,
        owner_prim=prim,
        body_path=body_path,
        reference_identity=reference_identity,
        kilograms_per_unit=kilograms_per_unit,
        meters_per_unit=meters_per_unit,
        UsdPhysics=UsdPhysics,
    )


def _authored_rigid_body_properties(prim: Any) -> tuple[str, ...]:
    authored_properties = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    return tuple(
        name for name in _RIGID_BODY_FALLBACK_PROPERTIES if name in authored_properties
    )


def _mass_api_evidence(prim: Any, *, UsdPhysics: Any) -> tuple[str, ...]:
    """Return composed MassAPI or authored MassAPI-property evidence."""

    evidence = ["PhysicsMassAPI"] if prim.HasAPI(UsdPhysics.MassAPI) else []
    evidence.extend(_authored_mass_properties(prim))
    return tuple(evidence)


def _authored_mass_properties(prim: Any) -> tuple[str, ...]:
    """Return authored standard MassAPI property specs in schema order."""

    authored_properties = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    return tuple(name for name in _MASS_API_PROPERTIES if name in authored_properties)


def _extract_mass(
    prim: Any,
    *,
    body_path: str,
    reference_identity: ArtifactIdentityV1,
    kilograms_per_unit: float,
    meters_per_unit: float,
    UsdPhysics: Any,
) -> MassPropertiesV1 | None:
    if not prim.HasAPI(UsdPhysics.MassAPI):
        authored_without_api = _authored_mass_properties(prim)
        if authored_without_api:
            _fail(
                "mass_property_without_api",
                f"{body_path} has MassAPI properties without applied MassAPI: "
                f"{list(authored_without_api)}",
            )
        return None
    unsupported = [
        name
        for name in _UNREPRESENTED_MASS_PROPERTIES
        if _is_authored_value_only_attribute(
            prim.GetAttribute(name),
            owner_path=body_path,
        )
    ]
    if unsupported:
        _fail(
            "unsupported_mass_property",
            f"{body_path} has unrepresented MassAPI opinions: {sorted(unsupported)}",
        )
    api = UsdPhysics.MassAPI(prim)
    mass_attr = api.GetMassAttr()
    inertia_attr = api.GetDiagonalInertiaAttr()
    center_attr = prim.GetAttribute("physics:centerOfMass")
    principal_attr = api.GetPrincipalAxesAttr()
    authored = (
        _is_authored_value_only_attribute(mass_attr, owner_path=body_path),
        _is_authored_value_only_attribute(inertia_attr, owner_path=body_path),
        _is_authored_value_only_attribute(center_attr, owner_path=body_path),
        _is_authored_value_only_attribute(principal_attr, owner_path=body_path),
    )
    if not any(authored):
        _fail(
            "incomplete_optional_schema",
            f"{body_path} has MassAPI without authored mass properties",
        )
    for attribute, present in zip(
        (mass_attr, inertia_attr, center_attr, principal_attr),
        authored,
        strict=True,
    ):
        if present:
            _require_static_attribute(attribute, owner_path=body_path)
    if not all(authored[:2]):
        required = ("physics:mass", "physics:diagonalInertia")
        authored_properties = _authored_mass_properties(prim)
        missing = [name for name in required if name not in authored_properties]
        _fail(
            "incomplete_optional_schema",
            f"{body_path} has incomplete MassAPI; required={list(required)!r}; "
            f"authored={list(authored_properties)!r}; missing={missing!r}",
        )
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (kilograms_per_unit, meters_per_unit)
    ):
        _fail("invalid_stage_units", "stage mass and length units must be positive")
    raw_mass = mass_attr.Get()
    raw_inertia = inertia_attr.Get()
    if raw_mass is None or raw_inertia is None:
        missing_values = [
            attribute.GetName()
            for attribute, value in (
                (mass_attr, raw_mass),
                (inertia_attr, raw_inertia),
            )
            if value is None
        ]
        _fail(
            "incomplete_optional_schema",
            f"{body_path} mass schema has no effective value for "
            f"{sorted(missing_values)}",
        )
    mass_stage_units = _required_finite_number(
        raw_mass,
        code="invalid_mass_properties",
        detail=f"{body_path} has invalid mass value {raw_mass!r}",
    )
    if mass_stage_units <= 0.0:
        _fail(
            "invalid_mass_properties",
            f"{body_path} mass must be positive; got {mass_stage_units!r}",
        )
    try:
        raw_inertia_values = tuple(raw_inertia)
    except TypeError:
        _fail(
            "invalid_mass_properties",
            f"{body_path} inertia is not a three-component value: {raw_inertia!r}",
        )
    if len(raw_inertia_values) != 3:
        _fail("invalid_mass_properties", f"{body_path} inertia must have 3 values")
    inertia_stage_units = tuple(
        _required_finite_number(
            value,
            code="invalid_mass_properties",
            detail=f"{body_path} has invalid inertia component {value!r}",
        )
        for value in raw_inertia_values
    )
    if any(value <= 0.0 for value in inertia_stage_units):
        _fail(
            "invalid_mass_properties",
            f"{body_path} inertia components must be positive: {inertia_stage_units!r}",
        )
    first, second, third = sorted(inertia_stage_units)
    if first + second < third and not math.isclose(
        first + second,
        third,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        _fail(
            "invalid_mass_properties",
            f"{body_path} inertia violates the inertia triangle: "
            f"{inertia_stage_units!r}",
        )
    mass_kg = mass_stage_units * kilograms_per_unit
    inertia_values = tuple(
        value * kilograms_per_unit * meters_per_unit * meters_per_unit
        for value in inertia_stage_units
    )
    if not math.isfinite(mass_kg) or any(
        not math.isfinite(value) for value in inertia_values
    ):
        _fail(
            "invalid_mass_properties",
            f"{body_path} mass conversion to SI produced non-finite values",
        )
    inertia = cast(tuple[float, float, float], inertia_values)
    center_of_mass = None
    principal_axes = None
    properties = [mass_attr.GetName(), inertia_attr.GetName()]
    if authored[2]:
        raw_center = center_attr.Get()
        if raw_center is None:
            _fail(
                "incomplete_optional_schema",
                f"{body_path} physics:centerOfMass has no effective value",
            )
        center_stage = _mass_vector3(
            raw_center,
            body_path=body_path,
            label="center of mass",
        )
        center_values = tuple(value * meters_per_unit for value in center_stage)
        if any(not math.isfinite(value) for value in center_values):
            _fail(
                "invalid_mass_properties",
                f"{body_path} center-of-mass conversion to SI produced "
                "non-finite values",
            )
        center_of_mass = cast(tuple[float, float, float], center_values)
        properties.append(center_attr.GetName())
    if authored[3]:
        quaternion = principal_attr.Get()
        if quaternion is None:
            _fail(
                "incomplete_optional_schema",
                f"{body_path} physics:principalAxes has no effective value",
            )
        try:
            imaginary = quaternion.GetImaginary()
            components = (
                quaternion.GetReal(),
                imaginary[0],
                imaginary[1],
                imaginary[2],
            )
        except (AttributeError, IndexError, TypeError):
            _fail(
                "invalid_mass_properties",
                f"{body_path} has invalid principal axes {quaternion!r}",
            )
        principal_axes = cast(
            tuple[float, float, float, float],
            tuple(
                _required_finite_number(
                    value,
                    code="invalid_mass_properties",
                    detail=(
                        f"{body_path} has invalid principal axes component {value!r}"
                    ),
                )
                for value in components
            ),
        )
        norm = math.sqrt(sum(value * value for value in principal_axes))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            _fail(
                "invalid_mass_properties",
                f"{body_path} principal axes must be normalized; got "
                f"{principal_axes!r}",
            )
        principal_axes = _canonicalize_quaternion(
            principal_axes,
            label=f"{body_path} physics:principalAxes",
        )
        properties.append(principal_attr.GetName())
    return MassPropertiesV1(
        mass_kg=mass_kg,
        center_of_mass_m=center_of_mass,
        diagonal_inertia_kg_m2=inertia,
        principal_axes=principal_axes,
        provenance=FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=body_path,
            properties=tuple(properties),
            derivation=(
                "stage_mass_and_length_units_to_si("
                "mass_kg=physics:mass*kilogramsPerUnit; "
                "center_of_mass_m=physics:centerOfMass*metersPerUnit when "
                "authored; diagonal_inertia_kg_m2=physics:diagonalInertia*"
                "kilogramsPerUnit*metersPerUnit^2)"
            ),
            evidence=f"Complete authored body-level mass properties on {body_path}.",
        ),
    )


def _extract_lifted_descendant_mass(
    prim: Any,
    *,
    owner_prim: Any,
    body_path: str,
    reference_identity: ArtifactIdentityV1,
    kilograms_per_unit: float,
    meters_per_unit: float,
    UsdPhysics: Any,
) -> MassPropertiesV1:
    contributor_path = str(prim.GetPath())
    try:
        mass = _extract_mass(
            prim,
            body_path=contributor_path,
            reference_identity=reference_identity,
            kilograms_per_unit=kilograms_per_unit,
            meters_per_unit=meters_per_unit,
            UsdPhysics=UsdPhysics,
        )
    except JointRiggerContractError as exc:
        if exc.code != "incomplete_optional_schema":
            raise
        required_properties = {
            "physics:mass",
            "physics:centerOfMass",
            "physics:diagonalInertia",
            "physics:principalAxes",
        }
        authored_properties = set(_authored_mass_properties(prim))
        raise JointRiggerContractError(
            "incomplete_descendant_mass_properties",
            f"mass contributor {contributor_path} is incomplete; "
            f"missing={sorted(required_properties - authored_properties)}: "
            f"{exc.detail}",
        ) from exc
    if mass is None:  # pragma: no cover - contributor discovery proves evidence
        _fail(
            "incomplete_descendant_mass_properties",
            f"mass contributor {contributor_path} has no effective MassAPI values",
        )
    assert mass is not None
    required_properties = {
        "physics:mass",
        "physics:centerOfMass",
        "physics:diagonalInertia",
        "physics:principalAxes",
    }
    actual_properties = set(mass.provenance.properties)
    if actual_properties != required_properties:
        _fail(
            "incomplete_descendant_mass_properties",
            f"mass contributor {contributor_path} must author exactly the complete "
            f"mass frame; missing={sorted(required_properties - actual_properties)}, "
            f"extra={sorted(actual_properties - required_properties)}",
        )
    assert mass.center_of_mass_m is not None
    assert mass.principal_axes is not None
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        _fail(
            "descendant_mass_contributor_not_collider",
            f"mass contributor {contributor_path} below owner {body_path} is not "
            "an authored CollisionAPI collider",
        )
    lifted = _lift_descendant_mass_frame(
        prim,
        owner_prim,
        center_of_mass_m=mass.center_of_mass_m,
        principal_axes=mass.principal_axes,
        meters_per_unit=meters_per_unit,
    )
    return MassPropertiesV1(
        mass_kg=mass.mass_kg,
        center_of_mass_m=lifted.center_of_mass_m,
        diagonal_inertia_kg_m2=mass.diagonal_inertia_kg_m2,
        principal_axes=lifted.principal_axes,
        provenance=FieldProvenanceV1(
            source="authored_reference",
            artifact=reference_identity,
            prim_path=contributor_path,
            properties=mass.provenance.properties,
            derivation=(
                f"{lifted.derivation_receipt}; "
                "stage_mass_and_length_units_to_si("
                "mass_kg=physics:mass*kilogramsPerUnit; "
                "center_of_mass_m=R*(physics:centerOfMass*metersPerUnit)+"
                "translation_stage*metersPerUnit; "
                "diagonal_inertia_kg_m2=physics:diagonalInertia*"
                "kilogramsPerUnit*metersPerUnit^2)"
            ),
            evidence=(
                f"Complete authored descendant collider mass properties on "
                f"{contributor_path}, lifted into rigid-body owner frame "
                f"{body_path}."
            ),
        ),
    )


def _mass_vector3(
    value: Any,
    *,
    body_path: str,
    label: str,
) -> tuple[float, float, float]:
    try:
        components = tuple(value)
    except TypeError:
        _fail(
            "invalid_mass_properties",
            f"{body_path} {label} is not a three-component value: {value!r}",
        )
    if len(components) != 3:
        _fail(
            "invalid_mass_properties",
            f"{body_path} {label} must have 3 values",
        )
    return cast(
        tuple[float, float, float],
        tuple(
            _required_finite_number(
                component,
                code="invalid_mass_properties",
                detail=(f"{body_path} has invalid {label} component {component!r}"),
            )
            for component in components
        ),
    )


def _colliders_for_body(
    *,
    owned_prims: Sequence[Any],
    source_stage: Any,
    reference_stage: Any,
    body_path: str,
    reference_identity: ArtifactIdentityV1,
    UsdPhysics: Any,
    UsdGeom: Any,
) -> tuple[ColliderPlanV1, ...]:
    colliders = []
    accepted_instance_root_paths: set[str] = set()
    for prim in owned_prims:
        path = str(prim.GetPath())
        if prim.IsInstanceProxy() and any(
            _is_path_under(path, instance_root_path)
            for instance_root_path in accepted_instance_root_paths
        ):
            # The exact instance-root target has already checked the complete
            # paired prototype, including proxy-authored properties. Proxy
            # descendants never become independent plan targets.
            continue
        _reject_unrepresented_relationships(
            prim,
            owner_path=path,
            relationship_names=("physics:simulationOwner",),
            unsupported_code="unsupported_collision_relationship",
            schema_label="CollisionAPI",
        )
        source_prim = source_stage.GetPrimAtPath(path)
        if _is_active_defined_prim(source_prim):
            _reject_unrepresented_relationships(
                source_prim,
                owner_path=path,
                relationship_names=("physics:simulationOwner",),
                unsupported_code="unsupported_collision_relationship",
                schema_label="source CollisionAPI",
            )
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        collision_evidence = _collision_api_evidence(prim, UsdPhysics=UsdPhysics)
        if not has_collision_api and collision_evidence:
            _fail(
                "collision_evidence_without_api",
                f"{path} has collision evidence without applied CollisionAPI: "
                f"{list(collision_evidence)}",
            )
        if not has_collision_api:
            continue
        _require_unrepresented_property_fallbacks(
            prim,
            owner_path=path,
            properties=_COLLISION_FALLBACK_PROPERTIES,
            unsupported_code="unsupported_collision_property",
            schema_label="CollisionAPI",
        )
        approximation: _ColliderApproximation | None = None
        properties = ["PhysicsCollisionAPI"]
        is_instance_root_xform = _is_instance_root_xform(prim, UsdGeom=UsdGeom)
        has_mesh_collision_api = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        approximation_attr = prim.GetAttribute("physics:approximation")
        approximation_authored = bool(
            approximation_attr
            and _is_authored_value_only_attribute(
                approximation_attr,
                owner_path=path,
            )
        )
        if approximation_authored and not has_mesh_collision_api:
            _fail(
                "mesh_collision_property_without_api",
                f"{path} has physics:approximation without MeshCollisionAPI",
            )
        if is_instance_root_xform and (
            not has_mesh_collision_api or not approximation_authored
        ):
            _fail(
                "instance_root_collider_evidence_incomplete",
                "Xform instance-root reference colliders require explicit "
                "PhysicsMeshCollisionAPI evidence and an authored approximation: "
                f"{path}",
            )
        if has_mesh_collision_api:
            properties.append("PhysicsMeshCollisionAPI")
            attribute = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
            if approximation_authored:
                _require_static_attribute(attribute, owner_path=path)
                raw_approximation = str(attribute.Get())
                if raw_approximation not in _SUPPORTED_COLLIDER_APPROXIMATIONS:
                    _fail(
                        "unsupported_collider_approximation",
                        f"{path} uses approximation {raw_approximation!r}",
                    )
                approximation = cast(_ColliderApproximation, raw_approximation)
                properties.append(attribute.GetName())
        _require_compatible_source_collider(
            prim,
            source_stage=source_stage,
            reference_stage=reference_stage,
            body_path=body_path,
            path=path,
            UsdGeom=UsdGeom,
        )
        if is_instance_root_xform:
            accepted_instance_root_paths.add(path)
        colliders.append(
            ColliderPlanV1(
                prim_path=path,
                mesh_collision_api=True if has_mesh_collision_api else None,
                mesh_approximation=approximation,
                provenance=_provenance(
                    reference_identity,
                    path,
                    tuple(properties),
                    f"Authored collider schema on {path}.",
                ),
            )
        )
    return tuple(colliders)


def _require_compatible_source_collider(
    reference_prim: Any,
    *,
    source_stage: Any,
    reference_stage: Any,
    body_path: str,
    path: str,
    UsdGeom: Any,
) -> None:
    """Require an emitted reference collider to target compatible source geometry."""

    source_prim = source_stage.GetPrimAtPath(path)
    if not source_prim or not source_prim.IsValid():
        _fail("collider_not_in_source", f"collider path does not resolve: {path}")
    if not source_prim.IsActive():
        _fail("source_collider_inactive", f"source collider is inactive: {path}")
    if not source_prim.IsDefined():
        _fail("source_collider_undefined", f"source collider is undefined: {path}")
    if reference_prim.IsInstanceProxy() or source_prim.IsInstanceProxy():
        stages = []
        if reference_prim.IsInstanceProxy():
            stages.append("reference")
        if source_prim.IsInstanceProxy():
            stages.append("source")
        _fail(
            "unsupported_instance_proxy_physics",
            f"collider is an instance proxy in {stages} and cannot be authored "
            f"without reshaping: {path}",
        )

    reference_type = str(reference_prim.GetTypeName())
    source_type = str(source_prim.GetTypeName())
    reference_is_gprim = reference_prim.IsA(UsdGeom.Gprim)
    source_is_gprim = source_prim.IsA(UsdGeom.Gprim)
    reference_is_instance_root = _is_instance_root_xform(
        reference_prim,
        UsdGeom=UsdGeom,
    )
    source_is_instance_root = _is_instance_root_xform(
        source_prim,
        UsdGeom=UsdGeom,
    )
    if reference_is_gprim or source_is_gprim:
        if not (
            reference_is_gprim and source_is_gprim and source_type == reference_type
        ):
            _fail(
                "source_collider_type_mismatch",
                f"collider type differs at {path}: "
                f"reference={reference_type!r}, source={source_type!r}",
            )
        _require_matching_collider_geometry(
            source_prim,
            reference_prim,
            path=path,
        )
    elif reference_is_instance_root or source_is_instance_root:
        if not (reference_is_instance_root and source_is_instance_root):
            _fail(
                "source_collider_instance_composition_mismatch",
                "collider instance-root status differs at "
                f"{path}: reference={reference_is_instance_root}, "
                f"source={source_is_instance_root}",
            )
        _require_matching_instance_collider_composition(
            source_prim,
            reference_prim,
            path=path,
            UsdGeom=UsdGeom,
        )
    else:
        _fail(
            "source_collider_type_mismatch",
            f"collider type differs at {path}: "
            f"reference={reference_type!r}, source={source_type!r}",
        )
    _require_matching_collider_transforms(
        source_stage,
        reference_stage,
        body_path=body_path,
        path=path,
        UsdGeom=UsdGeom,
    )


def _is_instance_root_xform(prim: Any, *, UsdGeom: Any) -> bool:
    """Return whether ``prim`` is an exact editable Xform instance root."""

    return bool(
        prim
        and prim.IsValid()
        and prim.IsA(UsdGeom.Xform)
        and prim.IsInstance()
        and not prim.IsInstanceProxy()
    )


def _require_matching_instance_collider_composition(
    source_prim: Any,
    reference_prim: Any,
    *,
    path: str,
    UsdGeom: Any,
) -> None:
    """Require paired instance roots to compose the same static collider tree."""

    from pxr import Usd

    source_prototype = source_prim.GetPrototype()
    reference_prototype = reference_prim.GetPrototype()
    if not source_prototype or not reference_prototype:
        _fail(
            "source_collider_instance_composition_mismatch",
            f"collider instance prototype does not resolve in both stages: {path}",
        )

    def relative_prims(prototype: Any) -> dict[str, Any]:
        prototype_path = str(prototype.GetPath())
        return {
            str(prim.GetPath()).removeprefix(prototype_path) or "/": prim
            for prim in Usd.PrimRange.AllPrims(prototype)
        }

    source_prims = relative_prims(source_prototype)
    reference_prims = relative_prims(reference_prototype)
    if set(source_prims) != set(reference_prims):
        _fail(
            "source_collider_instance_composition_mismatch",
            f"collider instance prototype paths differ at {path}: "
            f"source={sorted(source_prims)}, reference={sorted(reference_prims)}",
        )

    has_source_geometry = False
    has_reference_geometry = False
    for relative_path in sorted(source_prims):
        source = source_prims[relative_path]
        reference = reference_prims[relative_path]
        source_state = (
            str(source.GetTypeName()),
            bool(source.IsActive()),
            bool(source.IsDefined()),
            bool(source.IsAbstract()),
            bool(source.IsInstance()),
            bool(source.IsInstanceable()),
        )
        reference_state = (
            str(reference.GetTypeName()),
            bool(reference.IsActive()),
            bool(reference.IsDefined()),
            bool(reference.IsAbstract()),
            bool(reference.IsInstance()),
            bool(reference.IsInstanceable()),
        )
        if source_state != reference_state or not _usd_values_equal(
            source.GetAllMetadata(),
            reference.GetAllMetadata(),
        ):
            _fail(
                "source_collider_instance_composition_mismatch",
                f"collider instance prototype prim differs at {path}[{relative_path}]",
            )
        if source.IsInstance() or reference.IsInstance():
            _fail(
                "source_collider_instance_composition_mismatch",
                "nested collider instances are outside the reference oracle "
                f"contract at {path}[{relative_path}]",
            )

        has_source_geometry = has_source_geometry or source.IsA(UsdGeom.Gprim)
        has_reference_geometry = has_reference_geometry or reference.IsA(UsdGeom.Gprim)
        prototype_path = f"{path}[{relative_path}]"
        _require_matching_collider_geometry(
            source,
            reference,
            path=prototype_path,
            source_prototype_path=source_prototype.GetPath(),
            reference_prototype_path=reference_prototype.GetPath(),
        )
        _require_matching_prototype_collider_transform(
            source,
            reference,
            path=prototype_path,
            UsdGeom=UsdGeom,
        )
        _require_matching_prototype_properties(
            source,
            reference,
            path=prototype_path,
            source_prototype_path=source_prototype.GetPath(),
            reference_prototype_path=reference_prototype.GetPath(),
        )

    if not has_source_geometry or not has_reference_geometry:
        _fail(
            "source_collider_instance_composition_mismatch",
            f"collider instance prototype has no paired GPrim geometry: {path}",
        )


def _require_matching_prototype_properties(
    source_prim: Any,
    reference_prim: Any,
    *,
    path: str,
    source_prototype_path: Any,
    reference_prototype_path: Any,
) -> None:
    """Require all composed prototype properties to have exact paired parity."""

    source_attributes = {
        str(item.GetName()): item for item in source_prim.GetAttributes()
    }
    reference_attributes = {
        str(item.GetName()): item for item in reference_prim.GetAttributes()
    }
    if set(source_attributes) != set(reference_attributes):
        _fail(
            "source_collider_instance_composition_mismatch",
            f"collider prototype attributes differ at {path}",
        )
    for name in sorted(source_attributes):
        source = source_attributes[name]
        reference = reference_attributes[name]
        source_times = tuple(float(value) for value in source.GetTimeSamples())
        reference_times = tuple(float(value) for value in reference.GetTimeSamples())
        if source_times != reference_times or not _matching_composed_attribute(
            source,
            reference,
            source_prototype_path=source_prototype_path,
            reference_prototype_path=reference_prototype_path,
        ):
            _fail(
                "source_collider_instance_composition_mismatch",
                f"collider prototype attribute differs at {path}.{name}",
            )
        for time in source_times:
            if not _usd_values_equal(source.Get(time), reference.Get(time)):
                _fail(
                    "source_collider_instance_composition_mismatch",
                    "collider prototype time sample differs at "
                    f"{path}.{name} time={time}",
                )

    def relationships(
        prim: Any,
        *,
        prototype_path: Any,
    ) -> dict[str, tuple[bool, tuple[tuple[str, str], ...], Any]]:
        return {
            str(item.GetName()): (
                bool(item.IsCustom()),
                tuple(
                    _prototype_path_comparison_key(
                        target,
                        prototype_path=prototype_path,
                    )
                    for target in item.GetTargets()
                ),
                item.GetAllAuthoredMetadata(),
            )
            for item in prim.GetRelationships()
        }

    if not _usd_values_equal(
        relationships(source_prim, prototype_path=source_prototype_path),
        relationships(reference_prim, prototype_path=reference_prototype_path),
    ):
        _fail(
            "source_collider_instance_composition_mismatch",
            f"collider prototype relationships differ at {path}",
        )


def _require_matching_prototype_collider_transform(
    source_prim: Any,
    reference_prim: Any,
    *,
    path: str,
    UsdGeom: Any,
) -> None:
    """Require one paired prototype prim to have the same static local frame."""

    from pxr import Usd

    source_xformable = UsdGeom.Xformable(source_prim)
    reference_xformable = UsdGeom.Xformable(reference_prim)
    if bool(source_xformable) != bool(reference_xformable):
        _fail(
            "source_collider_transform_mismatch",
            f"collider prototype transformability differs at {path}",
        )
    if not source_xformable:
        return
    _require_static_collider_xform(
        source_xformable,
        stage_label="source prototype",
        collider_path=path,
        owner_path=path,
    )
    _require_static_collider_xform(
        reference_xformable,
        stage_label="reference prototype",
        collider_path=path,
        owner_path=path,
    )
    source_local = source_xformable.GetLocalTransformation(Usd.TimeCode.Default())
    reference_local = reference_xformable.GetLocalTransformation(Usd.TimeCode.Default())
    if bool(source_xformable.GetResetXformStack()) != bool(
        reference_xformable.GetResetXformStack()
    ) or not _matrices_close(source_local, reference_local):
        _fail(
            "source_collider_transform_mismatch",
            f"collider prototype local transform differs at {path}",
        )


def _require_matching_collider_geometry(
    source_prim: Any,
    reference_prim: Any,
    *,
    path: str,
    source_prototype_path: Any | None = None,
    reference_prototype_path: Any | None = None,
) -> None:
    """Require source geometry to reproduce the golden collider exactly."""

    source_attributes = _collider_geometry_attributes(source_prim)
    reference_attributes = _collider_geometry_attributes(reference_prim)
    if set(source_attributes) != set(reference_attributes):
        _fail(
            "source_collider_geometry_mismatch",
            f"collider geometry attributes differ at {path}: "
            f"source={sorted(source_attributes)}, "
            f"reference={sorted(reference_attributes)}",
        )

    for name in sorted(source_attributes):
        source_attribute = source_attributes[name]
        reference_attribute = reference_attributes[name]
        source_times = tuple(
            float(value) for value in source_attribute.GetTimeSamples()
        )
        reference_times = tuple(
            float(value) for value in reference_attribute.GetTimeSamples()
        )
        if source_times or reference_times:
            _fail(
                "time_varying_collider_geometry",
                f"collider geometry is time-sampled at {path}.{name}: "
                f"source={source_times}, reference={reference_times}",
            )
        if not _matching_composed_attribute(
            source_attribute,
            reference_attribute,
            source_prototype_path=source_prototype_path,
            reference_prototype_path=reference_prototype_path,
        ):
            _fail(
                "source_collider_geometry_mismatch",
                f"collider geometry attribute differs at {path}.{name}",
            )


def _collider_geometry_attributes(prim: Any) -> dict[str, Any]:
    """Return composed Gprim attributes that can affect collision geometry."""

    attributes: dict[str, Any] = {}
    for attribute in prim.GetAttributes():
        name = str(attribute.GetName())
        if (
            name in _COLLIDER_NON_GEOMETRY_ATTRIBUTES
            or name.startswith("primvars:")
            or name.startswith("xformOp:")
            or name.startswith("physics:")
            or name.startswith("physx")
            or name.startswith("drive:")
            or name.startswith("state:")
        ):
            continue
        attributes[name] = attribute
    return attributes


def _prototype_path_comparison_key(
    path: Any,
    *,
    prototype_path: Any | None,
) -> tuple[str, str]:
    """Return a stable key for a path composed inside a generated prototype."""

    if (
        prototype_path is not None
        and path.IsAbsolutePath()
        and path.HasPrefix(prototype_path)
    ):
        return (
            "prototype-relative",
            str(path.MakeRelativePath(prototype_path)),
        )
    return ("verbatim", str(path))


def _matching_composed_attribute(
    source: Any,
    reference: Any,
    *,
    source_prototype_path: Any | None = None,
    reference_prototype_path: Any | None = None,
) -> bool:
    """Compare one static composed attribute, including connection identity."""

    return bool(
        source
        and source.IsValid()
        and reference
        and reference.IsValid()
        and str(source.GetTypeName()) == str(reference.GetTypeName())
        and bool(source.IsCustom()) == bool(reference.IsCustom())
        and str(source.GetVariability()) == str(reference.GetVariability())
        and tuple(
            _prototype_path_comparison_key(
                path,
                prototype_path=source_prototype_path,
            )
            for path in source.GetConnections()
        )
        == tuple(
            _prototype_path_comparison_key(
                path,
                prototype_path=reference_prototype_path,
            )
            for path in reference.GetConnections()
        )
        and _usd_values_equal(
            source.GetAllAuthoredMetadata(),
            reference.GetAllAuthoredMetadata(),
        )
        and _usd_values_equal(source.Get(), reference.Get())
    )


def _require_matching_collider_transforms(
    source_stage: Any,
    reference_stage: Any,
    *,
    body_path: str,
    path: str,
    UsdGeom: Any,
) -> None:
    """Require a static, matching local chain and world collider transform."""

    from pxr import Usd

    current_path = path
    while True:
        source_prim = source_stage.GetPrimAtPath(current_path)
        reference_prim = reference_stage.GetPrimAtPath(current_path)
        source_xformable = UsdGeom.Xformable(source_prim)
        reference_xformable = UsdGeom.Xformable(reference_prim)
        if bool(source_xformable) != bool(reference_xformable):
            _fail(
                "source_collider_transform_mismatch",
                f"collider transformability differs at {current_path} for {path}",
            )
        if source_xformable and reference_xformable:
            _require_static_collider_xform(
                source_xformable,
                stage_label="source",
                collider_path=path,
                owner_path=current_path,
            )
            _require_static_collider_xform(
                reference_xformable,
                stage_label="reference",
                collider_path=path,
                owner_path=current_path,
            )
            source_local = source_xformable.GetLocalTransformation(
                Usd.TimeCode.Default()
            )
            reference_local = reference_xformable.GetLocalTransformation(
                Usd.TimeCode.Default()
            )
            if bool(source_xformable.GetResetXformStack()) != bool(
                reference_xformable.GetResetXformStack()
            ) or not _matrices_close(source_local, reference_local):
                _fail(
                    "source_collider_transform_mismatch",
                    f"collider local transform differs at {current_path} for {path}",
                )
        if current_path == body_path:
            break
        reference_parent = reference_prim.GetParent()
        # Callers only pass colliders selected beneath body_path. Retain a
        # fail-closed guard if that internal ownership invariant ever regresses.
        if not reference_parent or reference_parent.IsPseudoRoot():  # pragma: no cover
            _fail(
                "source_collider_transform_mismatch",
                f"collider {path} is not beneath planned body {body_path}",
            )
        current_path = str(reference_parent.GetPath())

    source_world = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        source_stage.GetPrimAtPath(path)
    )
    reference_world = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    ).GetLocalToWorldTransform(reference_stage.GetPrimAtPath(path))
    if not _matrices_close(source_world, reference_world):
        _fail(
            "source_collider_transform_mismatch",
            f"collider world transform differs between paired stages: {path}",
        )


def _require_static_collider_xform(
    xformable: Any,
    *,
    stage_label: str,
    collider_path: str,
    owner_path: str,
) -> None:
    """Reject time samples in one effective collider transform stack."""

    attributes = [
        xformable.GetXformOpOrderAttr(),
        *(operation.GetAttr() for operation in xformable.GetOrderedXformOps()),
    ]
    time_samples = sorted(
        {float(time) for attribute in attributes for time in attribute.GetTimeSamples()}
    )
    if time_samples:
        _fail(
            "time_varying_collider_transform",
            f"collider {collider_path} has a time-sampled {stage_label} transform "
            f"at {owner_path}: {time_samples}",
        )


def _matrices_close(left: Any, right: Any) -> bool:
    """Compare two transform matrices using the oracle's frame tolerance."""

    return all(
        math.isclose(
            left_value,
            right_value,
            rel_tol=0.0,
            abs_tol=_FRAME_TOLERANCE,
        )
        for left_value, right_value in zip(
            _matrix_values(left),
            _matrix_values(right),
            strict=True,
        )
    )


def _collision_api_evidence(prim: Any, *, UsdPhysics: Any) -> tuple[str, ...]:
    evidence = (
        ["PhysicsMeshCollisionAPI"] if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else []
    )
    authored_properties = {str(prop.GetName()) for prop in prim.GetAuthoredProperties()}
    if prim.IsA(UsdPhysics.Joint):
        authored_properties.discard("physics:collisionEnabled")
    evidence.extend(
        sorted(
            name for name in _COLLISION_API_PROPERTIES if name in authored_properties
        )
    )
    return tuple(evidence)


def _unowned_collision_evidence(
    stage: Any,
    *,
    body_path: str,
    all_rigid_body_paths: set[str],
    UsdPhysics: Any,
) -> tuple[str, ...]:
    """Return collider evidence not owned by any nested rigid body."""

    evidence = []
    for prim in _traverse_all_prims(stage):
        if not _is_active_defined_prim(prim):
            continue
        path = str(prim.GetPath())
        if not _is_path_under(path, body_path):
            continue
        if _nearest_body_owner(path, all_rigid_body_paths) is not None:
            continue
        properties = list(_collision_api_evidence(prim, UsdPhysics=UsdPhysics))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            properties.insert(0, "PhysicsCollisionAPI")
        simulation_owner = prim.GetRelationship("physics:simulationOwner")
        if simulation_owner and simulation_owner.IsAuthored():
            properties.append("physics:simulationOwner")
        if properties:
            evidence.append(f"{path}: {sorted(set(properties))}")
    return tuple(evidence)


def _extract_articulation_root(
    stage: Any,
    *,
    source_stage: Any,
    body_paths: set[str],
    joint_paths: set[str],
    reference_identity: ArtifactIdentityV1,
    UsdPhysics: Any,
) -> ArticulationRootPlanV1 | None:
    roots: list[tuple[str, bool]] = []
    for prim in _traverse_all_prims(stage):
        if not _is_active_defined_prim(prim) or not prim.HasAPI(
            UsdPhysics.ArticulationRootAPI
        ):
            continue
        path = str(prim.GetPath())
        includes_body = any(_is_path_under(body_path, path) for body_path in body_paths)
        includes_joint = any(
            _is_path_under(joint_path, path) for joint_path in joint_paths
        )
        if includes_body or includes_joint:
            if prim.IsInstanceProxy():
                _fail(
                    "unsupported_instance_proxy_physics",
                    "articulation root is an instance proxy in the reference "
                    f"stage and cannot be replayed without reshaping: {path}",
                )
            roots.append((path, includes_body))
    if not roots:
        return None
    if len(roots) != 1:
        _fail(
            "contradictory_articulation_roots",
            f"found roots {sorted(path for path, _ in roots)}",
        )
    path, includes_body = roots[0]
    if not includes_body:
        _fail(
            "unsupported_joint_articulation_root",
            f"articulation root {path} is associated only through selected joints",
        )
    source_prim = source_stage.GetPrimAtPath(path)
    if not _is_active_defined_prim(source_prim):
        _fail(
            "articulation_root_not_in_source",
            f"articulation root path does not resolve in source: {path}",
        )
    if source_prim.IsInstanceProxy():
        _fail(
            "unsupported_instance_proxy_physics",
            "articulation root is an instance proxy in the source stage and "
            f"cannot be authored without reshaping: {path}",
        )
    return ArticulationRootPlanV1(
        prim_path=path,
        provenance=_provenance(
            reference_identity,
            path,
            ("PhysicsArticulationRootAPI",),
            f"Authored articulation-root API on {path}.",
        ),
    )


def _artifact_identity(
    path: Path,
    uri: str,
    stage: Any,
    *,
    root_sha256: str,
    dependency_records: Sequence[dict[str, str]] | None = None,
) -> ArtifactIdentityV1:
    if not uri.strip():
        _fail("invalid_artifact_identity", "artifact URI must not be blank")
    get_root_layer = getattr(stage, "GetRootLayer", None)
    stage_root_layer = get_root_layer() if callable(get_root_layer) else None
    records = [
        {
            "kind": "artifact_root",
            "locator": "$artifact",
            "sha256": root_sha256,
        }
    ]
    if dependency_records is not None:
        records.extend(dict(record) for record in dependency_records)
    else:
        for layer in stage.GetUsedLayers():
            if layer.anonymous:
                continue
            raw_locator = str(
                layer.resolvedPath or layer.realPath or getattr(layer, "identifier", "")
            )
            if not raw_locator:
                _fail(
                    "invalid_artifact_identity",
                    "a non-anonymous used layer has no stable locator",
                )
            candidate = Path(raw_locator).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if candidate.is_file():
                layer_sha256 = _file_sha256(
                    candidate,
                    code="dependency_artifact_missing",
                )
            else:
                layer_sha256 = hashlib.sha256(
                    layer.ExportToString().encode()
                ).hexdigest()
            records.append(
                {
                    "kind": (
                        "stage_root_layer"
                        if layer == stage_root_layer
                        else "used_layer"
                    ),
                    "locator": _canonical_layer_locator(
                        raw_locator,
                        artifact_path=path,
                    ),
                    "sha256": layer_sha256,
                }
            )
    payload = json.dumps(
        {
            "schema_version": _BUNDLE_SCHEMA,
            "dependencies": sorted(
                records,
                key=lambda record: (
                    record["kind"],
                    record["locator"],
                    record["sha256"],
                ),
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return ArtifactIdentityV1(
        uri=uri,
        root_sha256=root_sha256,
        dependency_bundle_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _fresh_usd_dependency_inventory(path: Path) -> tuple[Any, Any, Any]:
    """Compute dependencies only when cached layers still match disk.

    OpenUSD layers are process-global resources. ``ComputeAllDependencies`` may
    otherwise reuse a layer whose backing file was replaced after the layer was
    opened, combining current root bytes with a stale composition closure. The
    tempting repair is a forced reload, but that can discard an edit racing a
    dirty-state check. Instead, every cached layer is compared with a fresh
    anonymous disk read before and after enumeration. Dirty or stale cache state
    fails closed without mutating the shared layer. Validation repeats for newly
    discovered layers until the dependency closure reaches a fixed point.
    """

    from pxr import Ar, Sdf, UsdUtils

    try:
        root_layer = Sdf.Layer.FindOrOpen(str(path))
    except Exception as exc:
        raise JointRiggerContractError(
            "artifact_dependency_enumeration_failed",
            f"could not open USD dependency root {path}: {exc}",
        ) from exc
    if root_layer is None:
        _fail(
            "artifact_dependency_enumeration_failed",
            f"could not open USD dependency root {path}",
        )

    validated: dict[str, Any] = {}
    pending = [root_layer]
    for _ in range(_MAX_DEPENDENCY_VALIDATION_PASSES):
        ordered_pending = sorted(
            pending,
            key=lambda layer: _dependency_layer_validation_key(layer, Ar=Ar),
        )
        ordered_pending = [
            layer
            for layer in ordered_pending
            if not bool(getattr(layer, "anonymous", False))
        ]
        identifiers = []
        for layer in ordered_pending:
            identifier = str(getattr(layer, "identifier", ""))
            if not identifier:
                _fail(
                    "artifact_dependency_refresh_failed",
                    "a non-anonymous USD dependency layer has no identifier",
                )
            identifiers.append(identifier)
        _require_layers_current_for_read(
            ordered_pending,
            identifiers=identifiers,
            Sdf=Sdf,
        )
        validated.update(zip(identifiers, ordered_pending, strict=True))

        try:
            layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        except Exception as exc:
            raise JointRiggerContractError(
                "artifact_dependency_enumeration_failed",
                f"could not enumerate USD dependencies for {path}: {exc}",
            ) from exc
        _require_layers_current_for_read(
            list(validated.values()),
            identifiers=list(validated),
            Sdf=Sdf,
        )
        pending = [
            layer
            for layer in layers
            if not bool(getattr(layer, "anonymous", False))
            and str(getattr(layer, "identifier", "")) not in validated
        ]
        if not pending:
            return layers, assets, unresolved

    raise JointRiggerContractError(
        "artifact_dependency_refresh_failed",
        "USD dependency cache validation did not converge after "
        f"{_MAX_DEPENDENCY_VALIDATION_PASSES} passes for {path}",
    )


def _require_layers_current_for_read(
    layers: Sequence[Any],
    *,
    identifiers: Sequence[str],
    Sdf: Any,
) -> None:
    """Fail when cached layers are dirty or differ from fresh disk reads."""

    if len(layers) != len(identifiers):  # pragma: no cover - internal guard
        _fail(
            "artifact_dependency_refresh_failed",
            "USD dependency validation layer and identifier counts differ",
        )
    for layer, identifier in zip(layers, identifiers, strict=True):
        if bool(getattr(layer, "dirty", False)):
            _fail(
                "artifact_dependency_cache_dirty",
                "refusing to read through unsaved edits in cached USD dependency "
                f"layer: {identifier}",
            )
        try:
            fresh = Sdf.Layer.OpenAsAnonymous(identifier)
        except Exception as exc:
            raise JointRiggerContractError(
                "artifact_dependency_refresh_failed",
                f"could not read fresh USD dependency layer {identifier}: {exc}",
            ) from exc
        if fresh is None:
            _fail(
                "artifact_dependency_refresh_failed",
                f"could not read fresh USD dependency layer: {identifier}",
            )
        if bool(getattr(layer, "dirty", False)):
            _fail(
                "artifact_dependency_cache_dirty",
                "cached USD dependency layer became dirty while establishing a "
                f"fresh read: {identifier}",
            )
        try:
            cached_text = layer.ExportToString()
            fresh_text = fresh.ExportToString()
        except Exception as exc:
            raise JointRiggerContractError(
                "artifact_dependency_refresh_failed",
                f"could not compare cached USD dependency layer {identifier} "
                f"with disk: {exc}",
            ) from exc
        if bool(getattr(layer, "dirty", False)):
            _fail(
                "artifact_dependency_cache_dirty",
                "cached USD dependency layer became dirty while comparing its "
                f"fresh read: {identifier}",
            )
        if cached_text != fresh_text:
            _fail(
                "artifact_dependency_cache_stale",
                "cached USD dependency layer differs from disk; release cached "
                f"stages or retry in a fresh process: {identifier}",
            )


def _dependency_layer_validation_key(layer: Any, *, Ar: Any) -> tuple[int, str]:
    """Validate package containers before their package-relative entries."""

    locator = str(
        getattr(layer, "resolvedPath", None)
        or getattr(layer, "realPath", None)
        or getattr(layer, "identifier", "")
    )
    outer = locator
    package_depth = 0
    while Ar.IsPackageRelativePath(outer):
        outer, _ = Ar.SplitPackageRelativePathOuter(outer)
        package_depth += 1
    return package_depth, locator


def _enumerate_usd_dependencies(
    path: Path,
    *,
    projection: _UsdCompositionProjection | None = None,
    root_mutated_code: str = "artifact_mutated",
    dependency_mutated_code: str = "artifact_dependency_mutated",
) -> tuple[_ResolvedUsdDependency, ...]:
    """Resolve the complete authored USD dependency inventory."""

    from pxr import Ar, Sdf

    if projection is None:
        with _usd_composition_projection((path,)) as owned_projection:
            return _enumerate_usd_dependencies(
                path,
                projection=owned_projection,
                root_mutated_code=root_mutated_code,
                dependency_mutated_code=dependency_mutated_code,
            )

    composition_path = projection.projected_path(path)
    layers, assets, unresolved = _fresh_usd_dependency_inventory(composition_path)
    projection.require_unchanged(
        path,
        code=dependency_mutated_code,
        root_code=root_mutated_code,
    )
    unresolved_values = sorted(
        {
            projection.original_identifier(str(value), Ar=Ar, Sdf=Sdf)
            for value in unresolved
        }
    )
    if unresolved_values:
        raise JointRiggerContractError(
            "unresolved_artifact_dependency",
            f"USD dependency closure contains unresolved paths: {unresolved_values}",
            unresolved_dependency_paths=tuple(unresolved_values),
        )

    dependencies: dict[tuple[str, str], _ResolvedUsdDependency] = {}
    for layer in layers:
        identifier = str(
            layer.resolvedPath or layer.realPath or getattr(layer, "identifier", "")
        )
        original_identifier = projection.original_identifier(
            identifier,
            Ar=Ar,
            Sdf=Sdf,
        )
        dependency = _resolved_usd_dependency(
            "layer",
            original_identifier,
            artifact_path=path,
            Ar=Ar,
            read_identifier=identifier,
            projection=projection,
            Sdf=Sdf,
        )
        dependencies[(dependency.kind, dependency.identifier)] = dependency
    for asset in assets:
        identifier = str(
            getattr(asset, "resolvedPath", None)
            or getattr(asset, "path", None)
            or asset
        )
        original_identifier = projection.original_identifier(
            identifier,
            Ar=Ar,
            Sdf=Sdf,
        )
        dependency = _resolved_usd_dependency(
            "asset",
            original_identifier,
            artifact_path=path,
            Ar=Ar,
            read_identifier=identifier,
            projection=projection,
            Sdf=Sdf,
        )
        dependencies[(dependency.kind, dependency.identifier)] = dependency
    represented_lexical_paths = {
        dependency.lexical_path
        for dependency in dependencies.values()
        if dependency.lexical_path is not None
    }
    projection_root = Path(os.path.abspath(path.expanduser()))
    for opaque_path in sorted(
        projection.opaque_dependencies.get(projection_root, set()),
        key=lambda item: item.as_posix(),
    ):
        if opaque_path in represented_lexical_paths:
            continue
        record = projection.files[opaque_path]
        dependency = _ResolvedUsdDependency(
            kind="opaque_asset",
            identifier=str(opaque_path),
            lexical_path=opaque_path,
            local_path=record.backing_path,
            package_relative=False,
            read_identifier=str(record.projected_path),
            captured_sha256=record.sha256,
        )
        dependencies[(dependency.kind, dependency.identifier)] = dependency
    return tuple(
        dependencies[key]
        for key in sorted(dependencies, key=lambda item: (item[0], item[1]))
    )


def _resolved_usd_dependency(
    kind: Literal["layer", "asset"],
    identifier: str,
    *,
    artifact_path: Path,
    Ar: Any,
    read_identifier: str | None = None,
    projection: _UsdCompositionProjection | None = None,
    Sdf: Any | None = None,
) -> _ResolvedUsdDependency:
    if not identifier:
        _fail("invalid_artifact_identity", f"a resolved {kind} has no identifier")
    # Inventory identifiers are already resolved inside the private projection.
    # Mapping them back to their original spelling must not resolve or stat the
    # live path again: it may have become a FIFO or another special file after
    # the retained regular-file copy was built.
    resolved = (
        identifier
        if read_identifier is not None
        else str(Ar.GetResolver().Resolve(identifier)) or identifier
    )
    authored_path = resolved
    if Sdf is not None:
        authored_path, _ = Sdf.Layer.SplitIdentifier(resolved)
    package_relative = bool(Ar.IsPackageRelativePath(authored_path))
    outer = authored_path
    while Ar.IsPackageRelativePath(outer):
        outer, _ = Ar.SplitPackageRelativePathOuter(outer)
    file_uri_path = _canonical_local_file_uri_path(
        outer,
        code="dependency_artifact_invalid",
        label=f"resolved USD {kind}",
    )
    if file_uri_path is not None:
        outer = str(file_uri_path)
    lexical_path = None
    local_path = None
    captured_sha256 = None
    if "://" not in outer and not _is_remote_resolver_locator(outer):
        candidate = Path(outer)
        if not candidate.is_absolute():
            candidate = artifact_path.parent / candidate
        lexical_path = Path(os.path.abspath(candidate))
        projected_record = (
            projection.files.get(lexical_path) if projection is not None else None
        )
        if read_identifier is not None and projection is not None:
            if projected_record is None:
                _fail(
                    "dependency_artifact_missing",
                    "resolved USD dependency is outside the retained private "
                    f"projection: {lexical_path}",
                )
            assert projected_record is not None
            local_path = projected_record.backing_path
            captured_sha256 = projected_record.sha256
        else:
            local_path = lexical_path.resolve(strict=False)
        if projected_record is None and not local_path.is_file():
            _fail(
                "dependency_artifact_missing",
                f"resolved USD {kind} is not a file: {local_path}",
            )
    return _ResolvedUsdDependency(
        kind=kind,
        identifier=resolved,
        lexical_path=lexical_path,
        local_path=local_path,
        package_relative=package_relative,
        read_identifier=read_identifier,
        captured_sha256=captured_sha256,
    )


def _dependency_identity_records(
    artifact_path: Path,
    dependencies: Sequence[_ResolvedUsdDependency],
) -> tuple[dict[str, str], ...]:
    artifact = artifact_path.expanduser().resolve(strict=False)
    records = []
    for dependency in dependencies:
        kind: str = dependency.kind
        if (
            dependency.kind == "layer"
            and not dependency.package_relative
            and dependency.local_path == artifact
        ):
            kind = "stage_root_layer"
        elif dependency.kind == "layer":
            kind = "used_layer"
        records.append(
            {
                "kind": kind,
                "locator": _canonical_layer_locator(
                    dependency.identifier,
                    artifact_path=artifact_path,
                ),
                "sha256": _resolved_dependency_sha256(dependency),
            }
        )
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record["kind"],
                record["locator"],
                record["sha256"],
            ),
        )
    )


def _resolved_dependency_sha256(dependency: _ResolvedUsdDependency) -> str:
    if dependency.local_path is not None and not dependency.package_relative:
        if dependency.captured_sha256 is not None:
            return dependency.captured_sha256
        return _file_sha256(
            dependency.local_path,
            code="dependency_artifact_missing",
        )

    from pxr import Ar

    read_identifier = dependency.read_identifier or dependency.identifier
    resolved = Ar.ResolvedPath(read_identifier)
    asset = Ar.GetResolver().OpenAsset(resolved)
    if asset is None:
        _fail(
            "dependency_artifact_missing",
            f"could not open resolved USD {dependency.kind}: {dependency.identifier}",
        )
    return _ar_asset_sha256(asset, identifier=dependency.identifier)


def _ar_asset_sha256(asset: Any, *, identifier: str) -> str:
    """Hash one resolver asset with bounded reads and exact size accounting."""

    try:
        size = int(asset.GetSize())
    except Exception as exc:
        raise JointRiggerContractError(
            "dependency_artifact_read_failed",
            f"could not determine resolved dependency size for {identifier}: {exc}",
        ) from exc
    if size < 0:
        _fail(
            "dependency_artifact_read_failed",
            f"resolved dependency reports a negative size for {identifier}: {size}",
        )

    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        count = min(_AR_ASSET_READ_CHUNK_SIZE, size - offset)
        try:
            chunk = bytes(asset.Read(count, offset))
        except Exception as exc:
            raise JointRiggerContractError(
                "dependency_artifact_read_failed",
                f"could not read resolved dependency {identifier} at offset "
                f"{offset}: {exc}",
            ) from exc
        if len(chunk) != count:
            _fail(
                "dependency_artifact_read_failed",
                f"short read for resolved dependency {identifier} at offset "
                f"{offset}: expected {count} bytes, got {len(chunk)}",
            )
        digest.update(chunk)
        offset += count
    return digest.hexdigest()


def _canonical_layer_locator(raw_locator: str, *, artifact_path: Path) -> str:
    """Return a deterministic locator without a relocatable absolute prefix."""

    from pxr import Ar, Sdf

    authored_path, arguments = Sdf.Layer.SplitIdentifier(raw_locator)
    package_path: str | None = None
    outer_locator = authored_path
    if Ar.IsPackageRelativePath(authored_path):
        outer_locator, package_path = Ar.SplitPackageRelativePathOuter(authored_path)
    file_uri_path = _canonical_local_file_uri_path(
        outer_locator,
        code="invalid_artifact_identity",
        label="USD dependency locator",
    )
    if file_uri_path is not None:
        outer_path = file_uri_path
    elif "://" in outer_locator or _is_remote_resolver_locator(outer_locator):
        # Resolver identifiers are not filesystem paths: in particular, their
        # dot segments are resolver-owned syntax and must not be collapsed.
        return raw_locator
    else:
        outer_path = Path(outer_locator)

    artifact = Path(os.path.abspath(artifact_path.expanduser()))
    if not outer_path.is_absolute():
        outer_path = artifact.parent / outer_path
    outer_path = Path(os.path.abspath(outer_path))
    if outer_path == artifact:
        canonical_outer = "$artifact"
    else:
        try:
            relative = os.path.relpath(outer_path, start=artifact.parent)
        except ValueError:  # pragma: no cover - cross-volume native Windows path
            canonical_outer = outer_path.as_posix()
        else:
            canonical_outer = Path(relative).as_posix()
    canonical = canonical_outer
    if package_path is not None:
        canonical = str(Ar.JoinPackageRelativePath(canonical, package_path))
    if arguments:
        canonical = str(Sdf.Layer.CreateIdentifier(canonical, arguments))
    return canonical


def _require_artifact_identity_unchanged(
    path: Path,
    stage: Any,
    expected: ArtifactIdentityV1,
    *,
    dependencies: Sequence[_ResolvedUsdDependency],
    projection: _UsdCompositionProjection,
    missing_code: str,
    root_mutated_code: str,
    dependency_mutated_code: str,
) -> None:
    """Fail when a root or any used layer changes after identity capture."""

    current_root_sha256 = _file_sha256(path, code=missing_code)
    if current_root_sha256 != expected.root_sha256:
        _fail(root_mutated_code, f"USD root changed while reading {path}")
    projection.require_unchanged(
        path,
        code=dependency_mutated_code,
        root_code=root_mutated_code,
    )
    current = _artifact_identity(
        path,
        expected.uri,
        stage,
        root_sha256=current_root_sha256,
        dependency_records=_dependency_identity_records(path, dependencies),
    )
    if current.dependency_bundle_sha256 != expected.dependency_bundle_sha256:
        _fail(
            dependency_mutated_code,
            f"USD dependency closure changed while reading {path}",
        )


def _reject_unrepresented_joint_schemas(prim: Any, *, joint_path: str) -> None:
    # ArticulationRootAPI is rejected later with graph-aware root diagnostics.
    schemas = sorted(
        token
        for token in _applied_schema_tokens(prim)
        if token.startswith(("Physics", "Physx"))
        and token not in {_PHYSX_JOINT_SCHEMA, "PhysicsArticulationRootAPI"}
        and not token.startswith("PhysicsDriveAPI:")
    )
    if schemas:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} uses unrepresented joint schemas: {schemas}",
        )
    properties = sorted(
        name
        for prop in prim.GetAuthoredProperties()
        if _is_physics_property_name(name := str(prop.GetName()))
        and name not in _REPRESENTED_JOINT_PROPERTIES
        and not name.startswith(_DRIVE_PROPERTY_PREFIX)
    )
    if properties:
        _fail(
            "unsupported_optional_schema",
            f"{joint_path} has unrepresented joint properties: {properties}",
        )


def _applied_schema_tokens(prim: Any) -> tuple[str, ...]:
    """Return registered and raw API-schema tokens in composed order."""

    tokens: list[str] = []
    seen: set[str] = set()
    metadata = prim.GetMetadata("apiSchemas")
    if metadata is not None:
        for token in metadata.GetAppliedItems():
            value = str(token)
            if value not in seen:
                tokens.append(value)
                seen.add(value)
    # Registered schemas can be present even when a synthetic/fake prim does
    # not expose a composed list-op. Preserve their API order after the raw
    # metadata sequence without losing unregistered authored tokens.
    for token in prim.GetAppliedSchemas():
        value = str(token)
        if value not in seen:
            tokens.append(value)
            seen.add(value)
    return tuple(tokens)


def _open_stage(
    path: Path,
    *,
    Usd: Any,
    label: str,
    display_path: Path | None = None,
) -> Any:
    reported_path = display_path or path
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception as exc:
        raise JointRiggerContractError(
            f"{label}_stage_open_failed",
            f"could not open {reported_path}: {exc}",
        ) from exc
    if stage is None:
        _fail(f"{label}_stage_open_failed", f"could not open {reported_path}")
    return stage


def _file_sha256(path: Path, *, code: str) -> str:
    """Hash one stable regular inode without following or blocking on races."""

    try:
        expected = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise JointRiggerContractError(code, f"file not found: {path}: {exc}") from exc
    if not stat.S_ISREG(expected.st_mode):
        _fail(code, f"file is not a non-symlink regular file: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise JointRiggerContractError(code, f"could not open {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        expected_state = (
            expected.st_dev,
            expected.st_ino,
            expected.st_mode,
            expected.st_nlink,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        opened_state = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_state != expected_state:
            _fail(code, f"file changed before it was opened: {path}")
        digest = hashlib.sha256()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(
                descriptor,
                min(1024 * 1024, opened.st_size - offset),
                offset,
            )
            if not chunk:
                _fail(code, f"file changed while it was hashed: {path}")
            digest.update(chunk)
            offset += len(chunk)
        if os.pread(descriptor, 1, offset):
            _fail(code, f"file grew while it was hashed: {path}")
        after = os.fstat(descriptor)
        observed_path = os.stat(path, follow_symlinks=False)
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_state = (
            observed_path.st_dev,
            observed_path.st_ino,
            observed_path.st_mode,
            observed_path.st_nlink,
            observed_path.st_size,
            observed_path.st_mtime_ns,
            observed_path.st_ctime_ns,
        )
        if after_state != expected_state or path_state != expected_state:
            _fail(code, f"file changed while it was hashed: {path}")
        return digest.hexdigest()
    finally:
        owned_descriptor = descriptor
        descriptor = -1
        os.close(owned_descriptor)


def _provenance(
    artifact: ArtifactIdentityV1,
    prim_path: str,
    properties: tuple[str, ...],
    evidence: str,
) -> FieldProvenanceV1:
    return FieldProvenanceV1(
        source="authored_reference",
        artifact=artifact,
        prim_path=prim_path,
        properties=properties,
        evidence=evidence,
    )


def _normalized_vector(value: Any, *, joint_path: str) -> tuple[float, float, float]:
    vector = (float(value[0]), float(value[1]), float(value[2]))
    if any(not math.isfinite(component) for component in vector):
        _fail("axis_not_finite", f"{joint_path} axis contains non-finite values")
    length = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(length) or math.isclose(length, 0.0, abs_tol=1e-12):
        _fail("axis_unresolved", f"{joint_path} axis cannot be normalized")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _validated_joint_frame_rotation(
    value: Any,
    *,
    joint_path: str,
    field_name: str,
    Gf: Any,
) -> Any:
    """Build a rotation only after validating its authored quaternion evidence."""

    try:
        imaginary = value.GetImaginary()
        components = (
            float(value.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        )
    except (AttributeError, IndexError, OverflowError, TypeError, ValueError):
        _fail(
            "invalid_joint_frame_rotation",
            f"{joint_path} {field_name} is not a valid quaternion: {value!r}",
        )
    if any(not math.isfinite(component) for component in components):
        _fail(
            "invalid_joint_frame_rotation",
            f"{joint_path} {field_name} contains non-finite quaternion components",
        )
    norm = math.hypot(*components)
    if norm <= _JOINT_FRAME_QUATERNION_ZERO_NORM_TOLERANCE:
        _fail(
            "invalid_joint_frame_rotation",
            f"{joint_path} {field_name} has a zero or near-zero quaternion norm",
        )
    if not math.isclose(
        norm,
        1.0,
        rel_tol=0.0,
        abs_tol=_JOINT_FRAME_QUATERNION_NORM_TOLERANCE,
    ):
        _fail(
            "invalid_joint_frame_rotation",
            f"{joint_path} {field_name} must be a unit quaternion; got norm {norm!r}",
        )
    try:
        return Gf.Rotation(value)
    except Exception as exc:
        _fail(
            "invalid_joint_frame_rotation",
            f"{joint_path} {field_name} could not construct a rotation: {exc}",
        )


def _finite_vector3(
    value: Any,
    *,
    code: str,
    detail: str,
) -> tuple[float, float, float]:
    try:
        vector = (float(value[0]), float(value[1]), float(value[2]))
    except (IndexError, OverflowError, TypeError, ValueError):
        _fail(code, detail)
    if any(not math.isfinite(component) for component in vector):
        _fail(code, detail)
    return vector


def _canonical_vector(value: Any) -> tuple[float, float, float]:
    return tuple(
        0.0
        if math.isclose(float(component), 0.0, abs_tol=1e-12)
        else round(float(component), 12)
        for component in value
    )  # type: ignore[return-value]


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


def _distance(left: Any, right: Any) -> float:
    return math.sqrt(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))
    )


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return math.nan
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return math.nan


def _required_finite_number(value: Any, *, code: str, detail: str) -> float:
    number = _optional_number(value)
    if number is None or not math.isfinite(number):
        _fail(code, detail)
    assert number is not None
    return number


def _nearest_body_owner(path: str, body_paths: set[str]) -> str | None:
    candidate = path.rstrip("/") or "/"
    while True:
        if candidate in body_paths:
            return candidate
        if candidate == "/":
            return None
        candidate = candidate.rpartition("/")[0] or "/"


def _prims_by_nearest_body_owner(
    prims: Iterable[Any],
    body_paths: set[str],
) -> dict[str, tuple[Any, ...]]:
    """Group active prims by their nearest rigid-body ancestor in one pass."""

    grouped: dict[str, list[Any]] = {}
    for prim in prims:
        owner = _nearest_body_owner(str(prim.GetPath()), body_paths)
        if owner is not None:
            grouped.setdefault(owner, []).append(prim)
    return {owner: tuple(owned_prims) for owner, owned_prims in grouped.items()}


def _is_path_under(path: str, root: str) -> bool:
    normalized = root.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def _fail(code: str, detail: str) -> None:
    raise JointRiggerContractError(code, detail)


__all__ = [
    "extract_reference_input",
    "identify_usd_artifact",
    "local_usd_dependency_paths",
    "write_reference_input",
]
