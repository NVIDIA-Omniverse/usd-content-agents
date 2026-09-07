# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-owned deterministic asset leaf discovery and runtime binding."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import inspect
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, Protocol, cast

from pydantic import BaseModel

from .models import (
    ASSET_LEAF_PROJECTOR_API_VERSION,
    ArtifactBinding,
    AssetLeafBundleIdentity,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjection,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafRegistrarIdentity,
    LeafReceiptArtifactCategory,
    canonical_asset_digest,
)

AssetLeafProjector = Callable[
    [BaseModel, BaseModel, AssetLeafProjectionContext],
    AssetLeafProjectionPayload,
]

ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP = (
    "content_agent_workflows.asset_leaf_runtime_bundles"
)
ASSET_LEAF_BUNDLE_DISTRIBUTION = "content-agent-workflows"
ASSET_LEAF_APPROVED_REGISTRAR_DISTRIBUTIONS = frozenset(
    {"content-agent-workflows", "joint-agent", "texture-agent"}
)
ASSET_LEAF_MANDATORY_REGISTRAR_DISTRIBUTIONS = frozenset(
    {ASSET_LEAF_BUNDLE_DISTRIBUTION}
)


class AssetLeafRuntimeBundleProvider(Protocol):
    """Repository-declared, side-effect-free loader for one domain bundle."""

    __module__: str
    __qualname__: str

    def __call__(self) -> AssetLeafRuntimeBundle: ...


def asset_model_schema_digest(model_type: type[BaseModel]) -> str:
    """Digest one exact public Pydantic schema used at a leaf boundary."""

    if not issubclass(model_type, BaseModel):
        raise TypeError("asset leaf schemas must be Pydantic BaseModel classes")
    return canonical_asset_digest(model_type.model_json_schema(mode="validation"))


def asset_projector_digest(
    projector_id: str,
    projector: AssetLeafProjector,
) -> str:
    """Bind a projector ID to its callable and containing module source."""

    try:
        source = textwrap.dedent(inspect.getsource(projector)).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"asset leaf projector source is unavailable: {projector_id}"
        ) from exc
    module = inspect.getmodule(projector)
    if module is None:
        raise ValueError(f"asset leaf projector module is unavailable: {projector_id}")
    try:
        module_source = inspect.getsource(module).replace("\r\n", "\n")
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"asset leaf projector module source is unavailable: {projector_id}"
        ) from exc
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    module_source_sha256 = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
    return canonical_asset_digest(
        {
            "projector_id": projector_id,
            "projector_api_version": ASSET_LEAF_PROJECTOR_API_VERSION,
            "implementation": f"{projector.__module__}.{projector.__qualname__}",
            "source_sha256": source_sha256,
            "module_source_sha256": module_source_sha256,
        }
    )


@dataclass(frozen=True)
class AssetLeafRuntimeBinding:
    """One descriptor resolved to exact schemas and a deterministic projector."""

    descriptor: AssetLeafDescriptor
    invocation_model: type[BaseModel]
    result_model: type[BaseModel]
    projector: AssetLeafProjector

    @classmethod
    def create(
        cls,
        *,
        leaf_id: str,
        entrypoint: str,
        invocation_model: type[BaseModel],
        result_model: type[BaseModel],
        projector_id: str,
        projector: AssetLeafProjector,
        required_artifact_categories: Iterable[LeafReceiptArtifactCategory] = (),
        required_dependencies: Iterable[str] = (),
        required_dependents: Iterable[str] = (),
        incompatible_leaf_ids: Iterable[str] = (),
    ) -> AssetLeafRuntimeBinding:
        """Create a self-consistent repository runtime registration."""

        descriptor = AssetLeafDescriptor.create(
            leaf_id=leaf_id,
            entrypoint=entrypoint,
            invocation_schema_digest=asset_model_schema_digest(invocation_model),
            result_schema_digest=asset_model_schema_digest(result_model),
            projection_schema_digest=asset_model_schema_digest(AssetLeafProjection),
            projector_id=projector_id,
            projector_digest=asset_projector_digest(projector_id, projector),
            required_artifact_categories=sorted(required_artifact_categories),
            required_dependencies=sorted(required_dependencies),
            required_dependents=sorted(required_dependents),
            incompatible_leaf_ids=sorted(incompatible_leaf_ids),
        )
        return cls(
            descriptor=descriptor,
            invocation_model=invocation_model,
            result_model=result_model,
            projector=projector,
        )

    def validate_identity(self) -> None:
        """Reject descriptor, schema, or projector drift at discovery/resolution."""

        expected = {
            "invocation_schema_digest": asset_model_schema_digest(
                self.invocation_model
            ),
            "result_schema_digest": asset_model_schema_digest(self.result_model),
            "projection_schema_digest": asset_model_schema_digest(AssetLeafProjection),
            "projector_digest": asset_projector_digest(
                self.descriptor.projector_id,
                self.projector,
            ),
        }
        for field, value in expected.items():
            if getattr(self.descriptor, field) != value:
                raise ValueError(
                    f"asset leaf {self.descriptor.leaf_id} {field} drifted"
                )

    def project(
        self,
        invocation: BaseModel,
        result: BaseModel,
        *,
        invocation_artifact: ArtifactBinding,
        result_artifact: ArtifactBinding,
    ) -> AssetLeafProjection:
        """Validate exact runtime types and execute the declared projector."""

        self.validate_identity()
        if type(invocation) is not self.invocation_model:
            raise TypeError(
                f"asset leaf {self.descriptor.leaf_id} invocation model drifted"
            )
        if type(result) is not self.result_model:
            raise TypeError(
                f"asset leaf {self.descriptor.leaf_id} result model drifted"
            )
        context = AssetLeafProjectionContext.create(
            leaf_id=self.descriptor.leaf_id,
            descriptor_digest=self.descriptor.descriptor_digest,
            invocation_schema_digest=self.descriptor.invocation_schema_digest,
            result_schema_digest=self.descriptor.result_schema_digest,
            projector_id=self.descriptor.projector_id,
            projector_api_version=ASSET_LEAF_PROJECTOR_API_VERSION,
            projector_digest=self.descriptor.projector_digest,
            context_schema_digest=asset_model_schema_digest(AssetLeafProjectionContext),
            invocation_artifact=invocation_artifact,
            result_artifact=result_artifact,
        )
        payload = self.projector(invocation, result, context)
        if type(payload) is not AssetLeafProjectionPayload:
            raise TypeError(
                f"asset leaf {self.descriptor.leaf_id} projector result drifted"
            )
        projected_categories = {
            category
            for category, bindings in (
                ("operation_index", payload.operation_indexes),
                ("evidence_index", payload.evidence_indexes),
                ("evidence", payload.evidence),
                ("saved_stage_readback", payload.saved_stage_readbacks),
                ("resource_release", payload.resource_release_receipts),
            )
            if bindings
        }
        missing_categories = sorted(
            set(self.descriptor.required_artifact_categories).difference(
                projected_categories
            )
        )
        if missing_categories:
            raise ValueError(
                f"asset leaf {self.descriptor.leaf_id} projector omitted required "
                f"artifact categories: {missing_categories}"
            )
        return AssetLeafProjection.create(
            leaf_id=self.descriptor.leaf_id,
            descriptor_digest=self.descriptor.descriptor_digest,
            invocation_schema_digest=self.descriptor.invocation_schema_digest,
            result_schema_digest=self.descriptor.result_schema_digest,
            projection_context_schema_digest=asset_model_schema_digest(
                AssetLeafProjectionContext
            ),
            projection_schema_digest=self.descriptor.projection_schema_digest,
            projector_id=self.descriptor.projector_id,
            projector_api_version=ASSET_LEAF_PROJECTOR_API_VERSION,
            projector_digest=self.descriptor.projector_digest,
            invocation=invocation_artifact,
            result=result_artifact,
            context=context,
            payload=payload,
        )


@dataclass(frozen=True)
class AssetLeafRuntimeBundle:
    """One domain-owned bundle discovered from repository package metadata."""

    bundle_id: str
    bindings: tuple[AssetLeafRuntimeBinding, ...]

    @classmethod
    def create(
        cls,
        *,
        bundle_id: str,
        bindings: Iterable[AssetLeafRuntimeBinding],
    ) -> AssetLeafRuntimeBundle:
        """Freeze a non-empty bundle; catalog composition owns final ordering."""

        normalized_id = bundle_id.strip()
        if not normalized_id or normalized_id != bundle_id:
            raise ValueError(
                "asset leaf runtime bundle ID must be stable and non-empty"
            )
        materialized = tuple(bindings)
        if not materialized:
            raise ValueError(f"asset leaf runtime bundle is empty: {bundle_id}")
        if any(
            type(binding) is not AssetLeafRuntimeBinding for binding in materialized
        ):
            raise TypeError(
                f"asset leaf runtime bundle contains an invalid binding: {bundle_id}"
            )
        return cls(bundle_id=normalized_id, bindings=materialized)


@dataclass(frozen=True)
class AssetLeafRuntimeRegistrar:
    """One approved installed distribution and its mandatory loaded bundles."""

    identity: AssetLeafRegistrarIdentity
    bundles: tuple[AssetLeafRuntimeBundle, ...]


@dataclass(frozen=True)
class AssetLeafRuntimeCatalog:
    """Immutable public catalog paired with repository runtime bindings."""

    catalog: AssetLeafCatalog
    bindings: Mapping[str, AssetLeafRuntimeBinding]

    def resolve(self, descriptor: AssetLeafDescriptor) -> AssetLeafRuntimeBinding:
        """Resolve only an exact descriptor frozen from this repository catalog."""

        binding = self.bindings.get(descriptor.leaf_id)
        if binding is None:
            raise ValueError(f"unknown repository asset leaf: {descriptor.leaf_id}")
        binding.validate_identity()
        if binding.descriptor != descriptor:
            raise ValueError(
                f"asset leaf descriptor identity drifted: {descriptor.leaf_id}"
            )
        return binding


def compose_asset_leaf_runtime_catalog(
    registrations: Iterable[AssetLeafRuntimeBinding],
    *,
    registrars: Iterable[AssetLeafRegistrarIdentity] = (),
) -> AssetLeafRuntimeCatalog:
    """Compose registrations deterministically without selecting or ordering a graph."""

    bindings: dict[str, AssetLeafRuntimeBinding] = {}
    for registration in registrations:
        if type(registration) is not AssetLeafRuntimeBinding:
            raise TypeError("asset leaf registration has an invalid runtime type")
        registration.validate_identity()
        leaf_id = registration.descriptor.leaf_id
        if leaf_id in bindings:
            raise ValueError(f"duplicate repository asset leaf registration: {leaf_id}")
        bindings[leaf_id] = registration
    if not bindings:
        raise ValueError("repository asset leaf catalog cannot be empty")
    descriptors = [bindings[leaf_id].descriptor for leaf_id in sorted(bindings)]
    catalog = AssetLeafCatalog.create(descriptors, registrars=list(registrars))
    for descriptor in catalog.descriptors:
        for dependency in descriptor.required_dependencies:
            dependency_descriptor = bindings[dependency].descriptor
            if (
                dependency in descriptor.incompatible_leaf_ids
                or descriptor.leaf_id in dependency_descriptor.incompatible_leaf_ids
            ):
                raise ValueError(
                    f"asset leaf {descriptor.leaf_id} requires incompatible "
                    f"registration {dependency}"
                )
        for dependent in descriptor.required_dependents:
            dependent_descriptor = bindings[dependent].descriptor
            if (
                dependent in descriptor.incompatible_leaf_ids
                or descriptor.leaf_id in dependent_descriptor.incompatible_leaf_ids
            ):
                raise ValueError(
                    f"asset leaf {descriptor.leaf_id} requires incompatible "
                    f"dependent registration {dependent}"
                )
    return AssetLeafRuntimeCatalog(
        catalog=catalog,
        bindings=MappingProxyType(
            {leaf_id: bindings[leaf_id] for leaf_id in sorted(bindings)}
        ),
    )


def compose_asset_leaf_runtime_bundles(
    bundles: Iterable[AssetLeafRuntimeBundle],
) -> AssetLeafRuntimeCatalog:
    """Compose domain-owned bundles without depending on declaration order."""

    by_id: dict[str, AssetLeafRuntimeBundle] = {}
    for bundle in bundles:
        if type(bundle) is not AssetLeafRuntimeBundle:
            raise TypeError("asset leaf bundle provider returned an invalid type")
        if bundle.bundle_id in by_id:
            raise ValueError(
                f"duplicate repository asset leaf bundle: {bundle.bundle_id}"
            )
        by_id[bundle.bundle_id] = bundle
    if not by_id:
        raise ValueError("repository asset leaf bundle set cannot be empty")
    return compose_asset_leaf_runtime_catalog(
        binding for bundle_id in sorted(by_id) for binding in by_id[bundle_id].bindings
    )


def compose_asset_leaf_runtime_registrars(
    registrars: Iterable[AssetLeafRuntimeRegistrar],
) -> AssetLeafRuntimeCatalog:
    """Compose approved registrars and bind every exact returned bundle identity."""

    by_id: dict[str, AssetLeafRuntimeRegistrar] = {}
    bundle_ids: set[str] = set()
    for registrar in registrars:
        if type(registrar) is not AssetLeafRuntimeRegistrar:
            raise TypeError("asset leaf registrar returned an invalid type")
        if type(registrar.identity) is not AssetLeafRegistrarIdentity:
            raise TypeError("asset leaf registrar identity has an invalid type")
        validated_identity = AssetLeafRegistrarIdentity.model_validate(
            registrar.identity.model_dump(mode="json")
        )
        if validated_identity != registrar.identity:  # pragma: no cover - defensive
            raise ValueError("asset leaf registrar identity changed during validation")
        if any(
            type(bundle) is not AssetLeafRuntimeBundle for bundle in registrar.bundles
        ):
            raise TypeError("asset leaf registrar contains an invalid bundle type")
        registrar_id = registrar.identity.registrar_id
        if registrar_id not in ASSET_LEAF_APPROVED_REGISTRAR_DISTRIBUTIONS:
            raise ValueError(
                f"unapproved repository asset leaf registrar: {registrar_id}"
            )
        if registrar_id in by_id:
            raise ValueError(
                f"duplicate repository asset leaf registrar: {registrar_id}"
            )
        if len(registrar.bundles) != len(registrar.identity.bundles):
            raise ValueError(f"asset leaf registrar is partial: {registrar_id}")
        runtime_bundles = {bundle.bundle_id: bundle for bundle in registrar.bundles}
        identity_bundles = {
            bundle.bundle_id: bundle for bundle in registrar.identity.bundles
        }
        if set(runtime_bundles) != set(identity_bundles):
            raise ValueError(
                f"asset leaf registrar bundle identity drifted: {registrar_id}"
            )
        for bundle_id, bundle in runtime_bundles.items():
            returned = {
                binding.descriptor.leaf_id: binding.descriptor.descriptor_digest
                for binding in bundle.bindings
            }
            if returned != identity_bundles[bundle_id].leaf_descriptor_digests:
                raise ValueError(
                    f"asset leaf registrar returned a stale bundle: "
                    f"{registrar_id}/{bundle_id}"
                )
        for bundle in registrar.bundles:
            if bundle.bundle_id in bundle_ids:
                raise ValueError(
                    f"duplicate repository asset leaf bundle: {bundle.bundle_id}"
                )
            bundle_ids.add(bundle.bundle_id)
        by_id[registrar_id] = registrar
    missing = sorted(ASSET_LEAF_MANDATORY_REGISTRAR_DISTRIBUTIONS.difference(by_id))
    if missing:
        raise RuntimeError(
            f"mandatory repository asset leaf registrars are missing: {missing}"
        )
    ordered = [by_id[registrar_id] for registrar_id in sorted(by_id)]
    return compose_asset_leaf_runtime_catalog(
        (
            binding
            for registrar in ordered
            for bundle in registrar.bundles
            for binding in bundle.bindings
        ),
        registrars=[registrar.identity for registrar in ordered],
    )


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _repository_asset_leaf_registrar_distributions() -> tuple[
    importlib_metadata.Distribution, ...
]:
    """Admit only approved installed distributions with explicit bundle metadata."""

    declared: dict[str, importlib_metadata.Distribution] = {}
    for distribution in importlib_metadata.distributions():
        entries = tuple(
            entry
            for entry in distribution.entry_points
            if entry.group == ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP
        )
        if not entries:
            continue
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("asset leaf registrar distribution has no identity")
        registrar_id = _normalized_distribution_name(raw_name)
        if registrar_id not in ASSET_LEAF_APPROVED_REGISTRAR_DISTRIBUTIONS:
            raise RuntimeError(
                f"unapproved asset leaf registrar distribution: {registrar_id}"
            )
        if registrar_id in declared:
            raise RuntimeError(
                f"asset leaf registrar distribution is duplicated: {registrar_id}"
            )
        declared[registrar_id] = distribution
    missing = sorted(ASSET_LEAF_MANDATORY_REGISTRAR_DISTRIBUTIONS.difference(declared))
    if missing:
        raise RuntimeError(
            f"mandatory repository asset leaf registrars are missing: {missing}"
        )
    return tuple(declared[name] for name in sorted(declared))


def _provider_source_identity(
    provider: AssetLeafRuntimeBundleProvider,
) -> tuple[str, str]:
    implementation = f"{provider.__module__}.{provider.__qualname__}"
    try:
        source = textwrap.dedent(inspect.getsource(cast(Any, provider))).replace(
            "\r\n", "\n"
        )
    except (OSError, TypeError) as exc:
        raise ValueError(
            f"asset leaf bundle provider source is unavailable: {implementation}"
        ) from exc
    return implementation, hashlib.sha256(source.encode("utf-8")).hexdigest()


def _load_repository_asset_leaf_runtime_registrars() -> tuple[
    AssetLeafRuntimeRegistrar, ...
]:
    """Load every approved declared bundle; one broken bundle fails discovery."""

    registrars: list[AssetLeafRuntimeRegistrar] = []
    for distribution in _repository_asset_leaf_registrar_distributions():
        registrar_id = _normalized_distribution_name(distribution.metadata["Name"])
        entries = sorted(
            (
                entry
                for entry in distribution.entry_points
                if entry.group == ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP
            ),
            key=lambda entry: entry.name,
        )
        entry_names = [entry.name for entry in entries]
        if len(entry_names) != len(set(entry_names)):
            raise RuntimeError(
                f"asset leaf bundle declarations are duplicated: {registrar_id}"
            )
        bundles: list[AssetLeafRuntimeBundle] = []
        identities: list[AssetLeafBundleIdentity] = []
        for entry in entries:
            try:
                loaded_provider = entry.load()
                if not callable(loaded_provider):
                    raise TypeError("bundle entry point is not callable")
                provider = cast(AssetLeafRuntimeBundleProvider, loaded_provider)
                implementation, source_sha256 = _provider_source_identity(provider)
                bundle = provider()
                if type(bundle) is not AssetLeafRuntimeBundle:
                    raise TypeError("bundle provider returned an invalid type")
                if bundle.bundle_id != entry.name:
                    raise ValueError(
                        "bundle provider identity differs from its package declaration"
                    )
                identity = AssetLeafBundleIdentity.create(
                    registrar_id=registrar_id,
                    bundle_id=bundle.bundle_id,
                    implementation=implementation,
                    source_sha256=source_sha256,
                    leaf_descriptor_digests={
                        binding.descriptor.leaf_id: binding.descriptor.descriptor_digest
                        for binding in bundle.bindings
                    },
                )
                if len(identity.leaf_descriptor_digests) != len(bundle.bindings):
                    raise ValueError("bundle returned duplicate leaf IDs")
            except Exception as exc:
                raise RuntimeError(
                    f"repository asset leaf bundle failed to load: "
                    f"{registrar_id}/{entry.name}"
                ) from exc
            bundles.append(bundle)
            identities.append(identity)
        registrar_identity = AssetLeafRegistrarIdentity.create(
            registrar_id=registrar_id,
            distribution_version=str(distribution.version),
            bundles=identities,
        )
        registrars.append(
            AssetLeafRuntimeRegistrar(
                identity=registrar_identity,
                bundles=tuple(bundles),
            )
        )
    return tuple(registrars)


@cache
def repository_asset_leaf_runtime_catalog() -> AssetLeafRuntimeCatalog:
    """Discover the sole deterministic catalog compiled into this repository."""

    return compose_asset_leaf_runtime_registrars(
        _load_repository_asset_leaf_runtime_registrars()
    )


def discover_repository_asset_leaf_catalog() -> AssetLeafCatalog:
    """Return the public capability catalog without making graph decisions."""

    return repository_asset_leaf_runtime_catalog().catalog


def resolve_repository_asset_leaf_catalog(
    candidate: AssetLeafCatalog,
) -> AssetLeafRuntimeCatalog:
    """Accept external catalog bytes only when they exactly resolve to repository IDs."""

    repository = repository_asset_leaf_runtime_catalog()
    if candidate != repository.catalog:
        raise ValueError(
            "external asset leaf catalog does not resolve to the repository catalog"
        )
    return repository


__all__ = [
    "ASSET_LEAF_APPROVED_REGISTRAR_DISTRIBUTIONS",
    "ASSET_LEAF_BUNDLE_DISTRIBUTION",
    "ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP",
    "ASSET_LEAF_MANDATORY_REGISTRAR_DISTRIBUTIONS",
    "AssetLeafRuntimeBinding",
    "AssetLeafRuntimeBundle",
    "AssetLeafRuntimeCatalog",
    "AssetLeafRuntimeRegistrar",
    "asset_model_schema_digest",
    "asset_projector_digest",
    "compose_asset_leaf_runtime_bundles",
    "compose_asset_leaf_runtime_catalog",
    "compose_asset_leaf_runtime_registrars",
    "discover_repository_asset_leaf_catalog",
    "repository_asset_leaf_runtime_catalog",
    "resolve_repository_asset_leaf_catalog",
]
