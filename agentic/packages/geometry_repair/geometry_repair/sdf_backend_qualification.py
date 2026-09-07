# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy-qualified SDF backends for Geometry Repair production routes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import sdf_tools

_POLICY_PATH = Path(__file__).with_name("sdf_backend_qualifications.json")
_POLICY_SCHEMA = "geometry-repair.sdf-backend-qualifications.v1"
_BACKEND_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICY_FIELDS = frozenset({"schema_version", "default_backend_id", "backends"})
_QUALIFICATION_FIELDS = frozenset(
    {
        "qualification_id",
        "required_operations",
        "expected_identity",
        "expected_provenance",
        "nonempty_provenance",
        "sha256_provenance",
    }
)
_IDENTITY_FIELDS = frozenset({"backend_id", "execution_mode", "implementation_version"})

SDF_REBUILD_BUILD_ID = "geometry-repair:sdf-rebuild:2"
SDF_REBUILD_IMPLEMENTATION_VERSION = "geometry-repair-sdf-rebuild-2"
GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS = frozenset(
    {
        sdf_tools.Operation.ACTIVE_VALUE_MASK,
        sdf_tools.Operation.EXTRACT_ENCLOSED_REGION,
        sdf_tools.Operation.FIELD_TO_MESH,
        sdf_tools.Operation.MESH_TO_SDF,
        sdf_tools.Operation.MESH_TO_UDF,
        sdf_tools.Operation.SMOOTH_SCALAR,
        sdf_tools.Operation.SMOOTH_SDF,
        sdf_tools.Operation.TOPOLOGY_TO_SDF,
    }
)


@dataclass(frozen=True, slots=True)
class SdfBackendQualification:
    """Behavioral qualification layered on an admitted ``sdf_tools`` driver."""

    backend_id: str
    qualification_id: str
    required_operations: frozenset[sdf_tools.Operation]
    expected_identity: Mapping[str, Any]
    expected_provenance: Mapping[str, Any]
    nonempty_provenance: frozenset[str]
    sha256_provenance: frozenset[str]

    def inspect(self) -> dict[str, Any]:
        """Load the admitted driver and verify its exact qualified identity."""

        session = sdf_tools.create_session(
            backend=self.backend_id,
            require=self.required_operations,
        )
        identity = session.backend_info.as_dict()
        self.validate(identity)
        return identity

    def validate(self, identity: Mapping[str, Any]) -> None:
        """Fail closed when runtime identity differs from qualification policy."""

        if any(
            not _strict_json_equal(identity.get(key), value)
            for key, value in self.expected_identity.items()
        ):
            raise sdf_tools.BackendUnavailableError(
                f"SDF backend {self.backend_id!r} identity differs from its Geometry Repair "
                "qualification"
            )
        operations = identity.get("operations")
        required = {operation.value for operation in self.required_operations}
        if not isinstance(operations, list) or not required.issubset(operations):
            raise sdf_tools.BackendUnavailableError(
                f"SDF backend {self.backend_id!r} no longer provides its qualified operations"
            )
        provenance = identity.get("provenance")
        if type(provenance) is not dict or any(
            not _strict_json_equal(provenance.get(key), value)
            for key, value in self.expected_provenance.items()
        ):
            raise sdf_tools.BackendUnavailableError(
                f"SDF backend {self.backend_id!r} provenance differs from its Geometry Repair "
                "qualification"
            )
        if any(
            not isinstance(provenance.get(key), str) or not provenance[key]
            for key in self.nonempty_provenance
        ):
            raise sdf_tools.BackendUnavailableError(
                f"SDF backend {self.backend_id!r} qualification provenance is incomplete"
            )
        if any(
            not isinstance(provenance.get(key), str) or _SHA256.fullmatch(provenance[key]) is None
            for key in self.sha256_provenance
        ):
            raise sdf_tools.BackendUnavailableError(
                f"SDF backend {self.backend_id!r} qualification digests are invalid"
            )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare a runtime JSON value without Python's bool/int coercions."""

    if isinstance(expected, Mapping):
        return (
            type(actual) is dict
            and actual.keys() == expected.keys()
            and all(_strict_json_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, tuple):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _string_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeError(f"{label} must be an object with string keys")
    return _freeze_json(value)


def _string_set(value: object, *, label: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise RuntimeError(f"{label} must be a list of nonempty strings")
    result = frozenset(value)
    if len(result) != len(value):
        raise RuntimeError(f"{label} must not contain duplicates")
    return result


def _load_policy() -> tuple[str, Mapping[str, SdfBackendQualification]]:
    try:
        payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read SDF backend qualification policy: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != _POLICY_FIELDS
        or payload.get("schema_version") != _POLICY_SCHEMA
    ):
        raise RuntimeError("unsupported SDF backend qualification policy")
    default_backend_id = payload.get("default_backend_id")
    raw_backends = payload.get("backends")
    if (
        not isinstance(default_backend_id, str)
        or _BACKEND_ID.fullmatch(default_backend_id) is None
        or not isinstance(raw_backends, dict)
        or not raw_backends
    ):
        raise RuntimeError("malformed SDF backend qualification policy")

    qualifications: dict[str, SdfBackendQualification] = {}
    for backend_id, raw in raw_backends.items():
        if (
            not isinstance(backend_id, str)
            or _BACKEND_ID.fullmatch(backend_id) is None
            or not isinstance(raw, dict)
            or set(raw) != _QUALIFICATION_FIELDS
        ):
            raise RuntimeError("malformed SDF backend qualification entry")
        qualification_id = raw.get("qualification_id")
        if not isinstance(qualification_id, str) or not qualification_id:
            raise RuntimeError(f"SDF backend {backend_id!r} has no qualification ID")
        raw_operations = _string_set(
            raw.get("required_operations"),
            label=f"SDF backend {backend_id!r} required_operations",
        )
        try:
            required_operations = frozenset(sdf_tools.Operation(item) for item in raw_operations)
        except ValueError as exc:
            raise RuntimeError(
                f"SDF backend {backend_id!r} qualification names an unknown operation"
            ) from exc
        if required_operations != GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS:
            raise RuntimeError(
                f"SDF backend {backend_id!r} qualification does not cover the canonical "
                "Geometry Repair operation set"
            )
        expected_identity = _string_mapping(
            raw.get("expected_identity"),
            label=f"SDF backend {backend_id!r} expected_identity",
        )
        if (
            set(expected_identity) != _IDENTITY_FIELDS
            or expected_identity.get("backend_id") != backend_id
        ):
            raise RuntimeError(
                f"SDF backend {backend_id!r} qualification has a mismatched identity"
            )
        qualifications[backend_id] = SdfBackendQualification(
            backend_id=backend_id,
            qualification_id=qualification_id,
            required_operations=required_operations,
            expected_identity=expected_identity,
            expected_provenance=_string_mapping(
                raw.get("expected_provenance"),
                label=f"SDF backend {backend_id!r} expected_provenance",
            ),
            nonempty_provenance=_string_set(
                raw.get("nonempty_provenance"),
                label=f"SDF backend {backend_id!r} nonempty_provenance",
            ),
            sha256_provenance=_string_set(
                raw.get("sha256_provenance"),
                label=f"SDF backend {backend_id!r} sha256_provenance",
            ),
        )
    if default_backend_id not in qualifications:
        raise RuntimeError("default SDF backend is not qualified")
    qualification_ids = [item.qualification_id for item in qualifications.values()]
    if len(qualification_ids) != len(set(qualification_ids)):
        raise RuntimeError("SDF backend qualification IDs must be unique")
    return default_backend_id, MappingProxyType(qualifications)


DEFAULT_SDF_BACKEND_ID, _QUALIFICATIONS = _load_policy()


def get_sdf_backend_qualification(backend_id: str) -> SdfBackendQualification:
    """Return one checked-in Geometry Repair backend qualification."""

    try:
        return _QUALIFICATIONS[backend_id]
    except KeyError as exc:
        raise sdf_tools.BackendUnavailableError(
            f"SDF backend {backend_id!r} is not qualified for Geometry Repair"
        ) from exc


def inspect_qualified_sdf_backend(backend_id: str) -> dict[str, Any]:
    """Return authenticated identity for one behaviorally qualified driver."""

    return get_sdf_backend_qualification(backend_id).inspect()


def qualified_sdf_backend_ids() -> tuple[str, ...]:
    """List backend IDs with checked-in Geometry Repair qualification evidence."""

    return tuple(sorted(_QUALIFICATIONS))


__all__ = [
    "DEFAULT_SDF_BACKEND_ID",
    "GEOMETRY_REPAIR_REQUIRED_SDF_OPERATIONS",
    "SDF_REBUILD_BUILD_ID",
    "SDF_REBUILD_IMPLEMENTATION_VERSION",
    "SdfBackendQualification",
    "get_sdf_backend_qualification",
    "inspect_qualified_sdf_backend",
    "qualified_sdf_backend_ids",
]
