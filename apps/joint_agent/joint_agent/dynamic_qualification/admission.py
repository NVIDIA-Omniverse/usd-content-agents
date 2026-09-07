# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ID-only admission of exact packaged dynamic qualification profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from importlib import resources
from threading import RLock
from typing import NoReturn, cast

from pydantic import ValidationError
from world_understanding.utils.captured_artifacts import (
    BoundedBinaryReader,
    CapturedArtifactError,
    CapturedOpaqueFile,
    OpaqueArtifactRequest,
    capture_streamed_opaque_file,
)

from joint_agent.capability_manifest import (
    LoadedCapabilityManifest,
    load_capability_manifest,
)
from joint_agent.dynamic_qualification.contracts import (
    ArtifactIdentityClaimV1,
    DynamicProfileRegistryEntryV1,
    DynamicProfileRegistryV1,
    DynamicQualificationProfileV1,
    require_canonical_token,
)

DYNAMIC_PROFILE_REGISTRY_RESOURCE = (
    "data/dynamic_qualification_profile_registry_v1.json"
)

_ADMISSION_SEAL = object()


class DynamicProfileAdmissionError(ValueError):
    """A requested profile did not cross the trusted admission boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class LoadedDynamicProfileRegistry:
    """Strict registry plus the exact packaged byte identity."""

    registry: DynamicProfileRegistryV1
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _AdmittedDynamicProfileAuthority:
    """Module-owned admission facts that caller mutation cannot rewrite."""

    profile_id: str
    identity: tuple[str, str, int]
    capture: CapturedOpaqueFile
    registry_sha256: str
    capability_manifest_version: str
    capability_manifest_sha256: str


class AdmittedDynamicProfile:
    """Context-bound profile proven against registry, manifest, and exact bytes.

    Normal construction is intentionally disabled. Instances are yielded only
    by :func:`admit_dynamic_profile` while the captured profile bytes remain
    open and immutable.
    """

    _capability_manifest_sha256: str
    _capability_manifest_version: str
    _capture: CapturedOpaqueFile
    _identity: ArtifactIdentityClaimV1
    _registry_sha256: str
    _seal: object

    __slots__ = (
        "_capability_manifest_sha256",
        "_capability_manifest_version",
        "_capture",
        "_identity",
        "_registry_sha256",
        "_seal",
    )

    def __new__(cls, *_args: object, **_kwargs: object) -> AdmittedDynamicProfile:
        raise TypeError("AdmittedDynamicProfile is created only by profile admission")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("admitted dynamic profiles are immutable")

    def __delattr__(self, _name: str) -> None:
        raise AttributeError("admitted dynamic profiles are immutable")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("AdmittedDynamicProfile cannot be subclassed")

    @property
    def profile(self) -> DynamicQualificationProfileV1:
        authority = _require_admitted_dynamic_profile(self)
        return _parse_captured_profile(authority.capture)

    @property
    def identity(self) -> ArtifactIdentityClaimV1:
        authority = _require_admitted_dynamic_profile(self)
        uri, sha256, size_bytes = authority.identity
        return ArtifactIdentityClaimV1.model_validate(
            {
                "uri": uri,
                "sha256": sha256,
                "size_bytes": size_bytes,
            },
            strict=True,
        )

    @property
    def registry_sha256(self) -> str:
        return _require_admitted_dynamic_profile(self).registry_sha256

    @property
    def capability_manifest_version(self) -> str:
        return _require_admitted_dynamic_profile(self).capability_manifest_version

    @property
    def capability_manifest_sha256(self) -> str:
        return _require_admitted_dynamic_profile(self).capability_manifest_sha256


_ACTIVE_ADMISSIONS: dict[
    AdmittedDynamicProfile,
    _AdmittedDynamicProfileAuthority,
] = {}
_ACTIVE_ADMISSIONS_LOCK = RLock()


def admit_dynamic_profile(
    profile_id: str,
) -> AbstractContextManager[AdmittedDynamicProfile]:
    """Admit one profile by ID from trusted packaged resources.

    There is deliberately no path, bytes, model, resolver, adapter, or manifest
    argument on this public boundary. Tests replace private trusted resource
    loaders rather than weakening this API.
    """

    return _admit_dynamic_profile(_require_profile_id(profile_id))


@contextmanager
def _admit_dynamic_profile(
    profile_id: str,
) -> Iterator[AdmittedDynamicProfile]:
    requested_id = _require_profile_id(profile_id)
    loaded_registry = _load_packaged_registry()
    registry = _strict_revalidate_registry(loaded_registry.registry)
    loaded_manifest = _load_capability_manifest()
    _validate_registry_manifest_bindings(registry, loaded_manifest)

    entry = next(
        (item for item in registry.profiles if item.profile_id == requested_id),
        None,
    )
    if entry is None:
        raise DynamicProfileAdmissionError(
            "profile_not_admitted",
            f"dynamic qualification profile is not admitted: {requested_id}",
        )

    request = OpaqueArtifactRequest(
        uri=entry.identity.uri,
        sha256=entry.identity.sha256,
        size_bytes=entry.identity.size_bytes,
        max_bytes=entry.max_bytes,
    )
    with _open_packaged_profile_resource(entry.resource) as stream:
        with capture_streamed_opaque_file(stream, request) as captured:
            try:
                profile = _parse_captured_profile(captured)
                _validate_profile_entry(profile, entry)
            except DynamicProfileAdmissionError:
                raise
            except (CapturedArtifactError, ValueError) as exc:
                raise DynamicProfileAdmissionError(
                    "profile_capture_failed",
                    f"admitted profile bytes could not be parsed safely: {exc}",
                ) from exc
            admitted = _create_admitted_dynamic_profile(
                identity=entry.identity,
                capture=captured,
                registry_sha256=loaded_registry.sha256,
                capability_manifest_version=(loaded_manifest.manifest.manifest_version),
                capability_manifest_sha256=loaded_manifest.sha256,
            )
            authority = _register_admitted_dynamic_profile(
                admitted,
                profile_id=profile.profile_id,
                identity=entry.identity,
                capture=captured,
                registry_sha256=loaded_registry.sha256,
                capability_manifest_version=(loaded_manifest.manifest.manifest_version),
                capability_manifest_sha256=loaded_manifest.sha256,
            )
            try:
                yield admitted
                captured.require_intact()
            finally:
                _unregister_admitted_dynamic_profile(admitted, authority)


def _load_packaged_registry() -> LoadedDynamicProfileRegistry:
    try:
        payload = (
            resources.files("joint_agent")
            .joinpath(DYNAMIC_PROFILE_REGISTRY_RESOURCE)
            .read_bytes()
        )
    except OSError as exc:
        raise DynamicProfileAdmissionError(
            "registry_unavailable",
            "dynamic qualification profile registry is unavailable",
        ) from exc
    _validate_strict_json(payload, "dynamic profile registry")
    try:
        registry = DynamicProfileRegistryV1.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise DynamicProfileAdmissionError(
            "registry_invalid",
            f"dynamic qualification profile registry is invalid: {exc}",
        ) from exc
    return LoadedDynamicProfileRegistry(
        registry=registry,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _load_capability_manifest() -> LoadedCapabilityManifest:
    return load_capability_manifest()


@contextmanager
def _open_packaged_profile_resource(resource: str) -> Iterator[BoundedBinaryReader]:
    try:
        stream = resources.files("joint_agent").joinpath(resource).open("rb")
    except OSError as exc:
        raise DynamicProfileAdmissionError(
            "profile_resource_unavailable",
            f"packaged dynamic qualification profile is unavailable: {resource}",
        ) from exc
    with stream:
        yield cast(BoundedBinaryReader, stream)


def _parse_captured_profile(
    captured: CapturedOpaqueFile,
) -> DynamicQualificationProfileV1:
    captured.require_intact()
    payload = bytearray()
    for chunk in captured.iter_chunks():
        payload.extend(chunk)
    captured.require_intact()
    raw_payload = bytes(payload)
    _validate_strict_json(raw_payload, "dynamic qualification profile")
    try:
        return DynamicQualificationProfileV1.model_validate_json(
            raw_payload,
            strict=True,
        )
    except ValidationError as exc:
        raise DynamicProfileAdmissionError(
            "profile_invalid",
            f"admitted dynamic qualification profile is invalid: {exc}",
        ) from exc


def _strict_revalidate_registry(
    registry: DynamicProfileRegistryV1,
) -> DynamicProfileRegistryV1:
    try:
        return DynamicProfileRegistryV1.model_validate(registry, strict=True)
    except ValidationError as exc:
        raise DynamicProfileAdmissionError(
            "registry_invalid",
            f"dynamic qualification profile registry is invalid: {exc}",
        ) from exc


def _validate_registry_manifest_bindings(
    registry: DynamicProfileRegistryV1,
    loaded_manifest: LoadedCapabilityManifest,
) -> None:
    manifest_bindings = {
        (row.capability_id, profile_id)
        for row in loaded_manifest.manifest.capabilities
        for profile_id in row.qualification.dynamic_profile_ids
    }
    registry_bindings = {
        (entry.capability_id, entry.profile_id) for entry in registry.profiles
    }
    if registry_bindings != manifest_bindings:
        missing = sorted(manifest_bindings - registry_bindings)
        extra = sorted(registry_bindings - manifest_bindings)
        raise DynamicProfileAdmissionError(
            "profile_manifest_binding_mismatch",
            "dynamic profile registry and capability manifest bindings disagree: "
            f"missing={missing}, extra={extra}",
        )


def _validate_profile_entry(
    profile: DynamicQualificationProfileV1,
    entry: DynamicProfileRegistryEntryV1,
) -> None:
    if profile.profile_id != entry.profile_id:
        raise DynamicProfileAdmissionError(
            "profile_id_mismatch",
            "captured profile ID disagrees with its registry entry",
        )
    if profile.capability_id != entry.capability_id:
        raise DynamicProfileAdmissionError(
            "profile_capability_mismatch",
            "captured profile capability disagrees with its registry entry",
        )


def _create_admitted_dynamic_profile(
    *,
    identity: ArtifactIdentityClaimV1,
    capture: CapturedOpaqueFile,
    registry_sha256: str,
    capability_manifest_version: str,
    capability_manifest_sha256: str,
) -> AdmittedDynamicProfile:
    admitted = object.__new__(AdmittedDynamicProfile)
    object.__setattr__(admitted, "_seal", _ADMISSION_SEAL)
    object.__setattr__(
        admitted,
        "_identity",
        ArtifactIdentityClaimV1.model_validate(
            identity.model_dump(mode="python", round_trip=True),
            strict=True,
        ),
    )
    object.__setattr__(admitted, "_capture", capture)
    object.__setattr__(admitted, "_registry_sha256", registry_sha256)
    object.__setattr__(
        admitted,
        "_capability_manifest_version",
        capability_manifest_version,
    )
    object.__setattr__(
        admitted,
        "_capability_manifest_sha256",
        capability_manifest_sha256,
    )
    return admitted


def _register_admitted_dynamic_profile(
    admitted: AdmittedDynamicProfile,
    *,
    profile_id: str,
    identity: ArtifactIdentityClaimV1,
    capture: CapturedOpaqueFile,
    registry_sha256: str,
    capability_manifest_version: str,
    capability_manifest_sha256: str,
) -> _AdmittedDynamicProfileAuthority:
    authority = _AdmittedDynamicProfileAuthority(
        profile_id=_require_profile_id(profile_id),
        identity=(identity.uri, identity.sha256, identity.size_bytes),
        capture=capture,
        registry_sha256=registry_sha256,
        capability_manifest_version=capability_manifest_version,
        capability_manifest_sha256=capability_manifest_sha256,
    )
    with _ACTIVE_ADMISSIONS_LOCK:
        if admitted in _ACTIVE_ADMISSIONS:  # pragma: no cover - internal invariant
            raise RuntimeError("admitted dynamic profile was registered twice")
        _ACTIVE_ADMISSIONS[admitted] = authority
    return authority


def _unregister_admitted_dynamic_profile(
    admitted: AdmittedDynamicProfile,
    authority: _AdmittedDynamicProfileAuthority,
) -> None:
    with _ACTIVE_ADMISSIONS_LOCK:
        registered = _ACTIVE_ADMISSIONS.pop(admitted, None)
    if registered is not authority:  # pragma: no cover - internal invariant
        raise RuntimeError("admitted dynamic profile authority was replaced")


def _active_admitted_dynamic_profile_authority(
    admitted: AdmittedDynamicProfile,
) -> _AdmittedDynamicProfileAuthority:
    """Resolve the live module-owned authority without trusting handle fields."""

    if not isinstance(admitted, AdmittedDynamicProfile):
        raise DynamicProfileAdmissionError(
            "profile_handle_untrusted",
            "dynamic qualification requires a live admitted profile handle",
        )
    with _ACTIVE_ADMISSIONS_LOCK:
        authority = _ACTIVE_ADMISSIONS.get(admitted)
    if authority is None:
        raise DynamicProfileAdmissionError(
            "profile_handle_untrusted",
            "dynamic qualification requires a live admitted profile handle",
        )
    return authority


def _require_admitted_dynamic_profile(
    admitted: AdmittedDynamicProfile,
) -> _AdmittedDynamicProfileAuthority:
    try:
        authority = _active_admitted_dynamic_profile_authority(admitted)
    except DynamicProfileAdmissionError:
        capture = getattr(admitted, "_capture", None)
        if (
            isinstance(admitted, AdmittedDynamicProfile)
            and getattr(admitted, "_seal", None) is _ADMISSION_SEAL
            and isinstance(capture, CapturedOpaqueFile)
        ):
            capture.require_intact()
        raise
    try:
        identity = admitted._identity
        identity_snapshot = (
            identity.uri,
            identity.sha256,
            identity.size_bytes,
        )
        handle_matches = (
            getattr(admitted, "_seal", None) is _ADMISSION_SEAL
            and admitted._capture is authority.capture
            and type(identity) is ArtifactIdentityClaimV1
            and identity_snapshot == authority.identity
            and admitted._registry_sha256 == authority.registry_sha256
            and (
                admitted._capability_manifest_version
                == authority.capability_manifest_version
            )
            and (
                admitted._capability_manifest_sha256
                == authority.capability_manifest_sha256
            )
        )
    except (AttributeError, TypeError):
        handle_matches = False
    if not handle_matches:
        raise DynamicProfileAdmissionError(
            "profile_handle_authority_mismatch",
            "admitted profile handle no longer matches its module-owned authority",
        )
    authority.capture.require_intact()
    return authority


def _captured_profile_file(
    admitted: AdmittedDynamicProfile,
) -> CapturedOpaqueFile:
    return _require_admitted_dynamic_profile(admitted).capture


def _require_profile_id(value: str) -> str:
    if not isinstance(value, str):
        raise DynamicProfileAdmissionError(
            "profile_id_invalid",
            "dynamic qualification profile ID must be a string",
        )
    try:
        return require_canonical_token(
            value,
            "dynamic qualification profile ID",
        )
    except ValueError as exc:
        raise DynamicProfileAdmissionError(
            "profile_id_invalid",
            "dynamic qualification profile ID must be canonical nonblank text",
        ) from exc


class _DuplicateJsonKeyError(ValueError):
    pass


def _validate_strict_json(payload: bytes, label: str) -> None:
    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise _DuplicateJsonKeyError(f"duplicate JSON key: {key}")
            document[key] = value
        return document

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        document = json.loads(
            payload,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DynamicProfileAdmissionError(
            "json_invalid",
            f"{label} must be strict duplicate-free finite JSON: {exc}",
        ) from exc
    if not isinstance(document, dict):
        raise DynamicProfileAdmissionError(
            "json_invalid",
            f"{label} root must be an object",
        )


__all__ = [
    "AdmittedDynamicProfile",
    "DYNAMIC_PROFILE_REGISTRY_RESOURCE",
    "DynamicProfileAdmissionError",
    "LoadedDynamicProfileRegistry",
    "admit_dynamic_profile",
]
