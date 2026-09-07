# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, fail-closed contracts for shared child-agent launches."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from ipaddress import IPv6Address
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from content_agent_workflows.common.artifacts import read_contained_artifact
from pydantic import BaseModel, ConfigDict, Field, model_validator

CHILD_LAUNCH_DESCRIPTOR_SCHEMA_VERSION: Literal["content-agents.child-launch.v1"] = (
    "content-agents.child-launch.v1"
)
CHILD_LAUNCH_ARTIFACT_CONTRACT_SCHEMA_VERSION: Literal[
    "content-agents.child-launch-artifacts.v1"
] = "content-agents.child-launch-artifacts.v1"
CHILD_LAUNCH_NETWORK_POLICY_SCHEMA_VERSION: Literal[
    "content-agents.child-launch-network-policy.v1"
] = "content-agents.child-launch-network-policy.v1"

ARTICULATION_CHILD_LAUNCH_PROFILE = "articulation.author"
TEXTURE_CHILD_LAUNCH_PROFILE = "texture.generate"
VALIDATION_CHILD_LAUNCH_PROFILE = "validation.plan"

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# These parent-owned values are not model-backend credentials registered in
# API_KEY_ENV_VAR_MAP, but they still grant domain or renderer authority and
# therefore must not enter a reasoning-only child.
_REASONING_ONLY_DOMAIN_AUTHORITY_ENVIRONMENT_NAMES = (
    "NGC_API_KEY",
    "NVCF_API_KEY",
    "NVCF_RENDER_FUNCTION_ID",
)


class ChildLaunchContractError(RuntimeError):
    """Raised before launch when a child contract is absent or inconsistent."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChildLaunchArtifactIdentity(_FrozenModel):
    """Exact identity of one parent-prepared, run-confined artifact."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ChildLaunchRunContract(_FrozenModel):
    """Confined run root and bridge artifact destinations for one turn."""

    schema_version: Literal["content-agents.child-launch-artifacts.v1"] = (
        CHILD_LAUNCH_ARTIFACT_CONTRACT_SCHEMA_VERSION
    )
    run_root: str = Field(min_length=1)
    child_output_path: str = Field(min_length=1)
    child_final_path: str = Field(min_length=1)
    bridge_artifact_prefix: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_confinement(self) -> ChildLaunchRunContract:
        root = Path(self.run_root)
        if not root.is_absolute():
            raise ValueError("child launch run_root must be absolute")
        for label, value in (
            ("child_output_path", self.child_output_path),
            ("child_final_path", self.child_final_path),
        ):
            candidate = Path(value)
            if not candidate.is_absolute():
                raise ValueError(f"child launch {label} must be absolute")
            if candidate == root or root not in candidate.parents:
                raise ValueError(f"child launch {label} must stay inside run_root")
        return self


class ChildLaunchNetworkPolicy(_FrozenModel):
    """Network authority granted to sandboxed child tool execution."""

    schema_version: Literal["content-agents.child-launch-network-policy.v1"] = (
        CHILD_LAUNCH_NETWORK_POLICY_SCHEMA_VERSION
    )
    mode: Literal["legacy_runtime", "reasoning_transport_only"]
    tool_network_access: bool
    allowed_hosts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_policy(self) -> ChildLaunchNetworkPolicy:
        if len(set(self.allowed_hosts)) != len(self.allowed_hosts):
            raise ValueError("child launch network hosts must be unique")
        for host in self.allowed_hosts:
            bracketed_ipv6 = host.startswith("[") and host.endswith("]")
            expected_hostname = host[1:-1] if bracketed_ipv6 else host
            if bracketed_ipv6:
                try:
                    IPv6Address(expected_hostname)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid child launch network host: {host!r}"
                    ) from exc
            try:
                parsed = urlsplit(f"//{host}")
                parsed_port = parsed.port
            except ValueError as exc:
                raise ValueError(
                    f"invalid child launch network host: {host!r}"
                ) from exc
            if (
                not host
                or parsed.hostname != expected_hostname
                or parsed_port is not None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(f"invalid child launch network host: {host!r}")
        if self.mode == "reasoning_transport_only":
            if self.tool_network_access:
                raise ValueError(
                    "reasoning-only child launches cannot grant unrestricted "
                    "tool network access"
                )
            if any(host not in {"127.0.0.1", "[::1]"} for host in self.allowed_hosts):
                raise ValueError(
                    "reasoning-only child launches can allow only loopback adapters"
                )
        return self


class ChildLaunchCredentialPolicy(_FrozenModel):
    """Domain credential names that must not enter the child environment."""

    mode: Literal["legacy_sanitized", "reasoning_transport_only"]
    forbidden_environment_names: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_environment_names(self) -> ChildLaunchCredentialPolicy:
        if len(set(self.forbidden_environment_names)) != len(
            self.forbidden_environment_names
        ):
            raise ValueError("forbidden child environment names must be unique")
        for name in self.forbidden_environment_names:
            if not _ENVIRONMENT_NAME.fullmatch(name):
                raise ValueError(f"invalid child environment name: {name!r}")
        return self


class ChildLaunchRunnerIdentity(_FrozenModel):
    """Resolved reasoning runner and model identity used for request construction."""

    runner: Literal["codex", "claude"]
    model: str | None = None
    model_reasoning_effort: str | None = None
    claude_execution_mode: Literal["sdk", "cli"] | None = None

    @model_validator(mode="after")
    def validate_runner_mode(self) -> ChildLaunchRunnerIdentity:
        if self.runner == "claude" and self.claude_execution_mode is None:
            raise ValueError("Claude child launch identity requires an execution mode")
        if self.runner == "codex" and self.claude_execution_mode is not None:
            raise ValueError("Codex child launch identity cannot select a Claude mode")
        return self


class ChildLaunchDescriptor(_FrozenModel):
    """One complete, provider-neutral child launch authorization."""

    schema_version: Literal["content-agents.child-launch.v1"] = (
        CHILD_LAUNCH_DESCRIPTOR_SCHEMA_VERSION
    )
    profile_key: str = Field(min_length=1)
    workflow: str = Field(min_length=1)
    workflow_skill: str = Field(min_length=1)
    scene_backend: str = Field(min_length=1)
    required_staged_skills: tuple[str, ...]
    skill_staging_mode: Literal["catalog", "required"]
    capability_inventory: ChildLaunchArtifactIdentity | None = None
    domain_policy_bounds: ChildLaunchArtifactIdentity | None = None
    network_policy: ChildLaunchNetworkPolicy
    credential_policy: ChildLaunchCredentialPolicy
    artifacts: ChildLaunchRunContract
    runner_identity: ChildLaunchRunnerIdentity


class ChildLaunchProfile(_FrozenModel):
    """Static registered identity and confinement policy for one config family."""

    key: str = Field(min_length=1)
    workflow: str = Field(min_length=1)
    workflow_skill: str = Field(min_length=1)
    scene_backend: str = Field(min_length=1)
    required_staged_skills: tuple[str, ...]
    skill_staging_mode: Literal["catalog", "required"] = "catalog"
    network_mode: Literal["legacy_runtime", "reasoning_transport_only"] = (
        "legacy_runtime"
    )
    requires_domain_contract: bool = False

    @model_validator(mode="after")
    def validate_profile(self) -> ChildLaunchProfile:
        if not self.required_staged_skills:
            raise ValueError("child launch profiles require at least one staged skill")
        if len(set(self.required_staged_skills)) != len(self.required_staged_skills):
            raise ValueError("child launch profile staged skills must be unique")
        if self.workflow_skill not in self.required_staged_skills:
            raise ValueError(
                "child launch workflow_skill must be included in required_staged_skills"
            )
        if self.network_mode == "reasoning_transport_only" and (
            self.skill_staging_mode != "required" or not self.requires_domain_contract
        ):
            raise ValueError(
                "reasoning-only profiles require exact skills and a domain contract"
            )
        return self


PRODUCTION_CHILD_LAUNCH_PROFILES = (
    ChildLaunchProfile(
        key="asset.run",
        workflow="asset.run",
        workflow_skill="content-workflow-asset",
        scene_backend="usd-cli",
        required_staged_skills=("content-workflow-asset",),
    ),
    ChildLaunchProfile(
        key="materials.assign",
        workflow="materials.assign",
        workflow_skill="content-workflow-material",
        scene_backend="usd-cli",
        required_staged_skills=("usd-cli", "content-workflow-material"),
    ),
    ChildLaunchProfile(
        key="physics.apply",
        workflow="physics.apply",
        workflow_skill="content-workflow-physics",
        scene_backend="usd-cli",
        required_staged_skills=("usd-cli", "content-workflow-physics"),
    ),
    ChildLaunchProfile(
        key="scene.run",
        workflow="scene.run",
        workflow_skill="content-workflow-large-scene",
        scene_backend="usd-cli",
        required_staged_skills=("content-workflow-large-scene",),
    ),
    ChildLaunchProfile(
        key="mesh-segmentation.run",
        workflow="mesh-segmentation.run",
        workflow_skill="content-workflow-mesh-segmentation",
        scene_backend="usd-cli",
        required_staged_skills=("content-workflow-mesh-segmentation",),
    ),
    ChildLaunchProfile(
        key="materials.author",
        workflow="materials.author",
        workflow_skill="content-workflow-material-authoring",
        scene_backend="usd-cli",
        required_staged_skills=("content-workflow-material-authoring",),
    ),
    ChildLaunchProfile(
        key="physics.refine_external.agentic",
        workflow="physics.refine_external.agentic",
        workflow_skill="content-workflow-physics-external-tuning",
        scene_backend="usd-cli",
        required_staged_skills=("content-workflow-physics-external-tuning",),
    ),
    ChildLaunchProfile(
        key=ARTICULATION_CHILD_LAUNCH_PROFILE,
        workflow="articulation.author",
        workflow_skill="content-workflow-articulation",
        scene_backend="usd-cli",
        required_staged_skills=(
            "usd-cli",
            "content-articulation-inspection",
            "content-articulation-proposal",
            "content-articulation-review",
            "content-articulation-authoring",
            "content-workflow-articulation",
        ),
        skill_staging_mode="required",
        network_mode="reasoning_transport_only",
        requires_domain_contract=True,
    ),
    ChildLaunchProfile(
        key=TEXTURE_CHILD_LAUNCH_PROFILE,
        workflow="texture.generate",
        workflow_skill="content-workflow-texture",
        scene_backend="usd-cli",
        required_staged_skills=(
            "usd-cli",
            "content-texture-scope",
            "content-texture-candidate",
            "content-texture-quality",
            "content-texture-publish",
            "content-workflow-texture",
        ),
        skill_staging_mode="required",
        network_mode="reasoning_transport_only",
        requires_domain_contract=True,
    ),
    ChildLaunchProfile(
        key=VALIDATION_CHILD_LAUNCH_PROFILE,
        workflow="validation.plan",
        workflow_skill="content-workflow-validation",
        scene_backend="none",
        required_staged_skills=("content-workflow-validation",),
        skill_staging_mode="required",
        network_mode="reasoning_transport_only",
        requires_domain_contract=True,
    ),
)


def build_child_launch_profile_registry(
    profiles: Iterable[ChildLaunchProfile],
) -> dict[str, ChildLaunchProfile]:
    """Build a unique profile registry and reject duplicate identities."""

    registry: dict[str, ChildLaunchProfile] = {}
    workflows: dict[str, str] = {}
    for profile in profiles:
        if profile.key in registry:
            raise ChildLaunchContractError(
                f"Duplicate child launch profile key: {profile.key}"
            )
        prior_key = workflows.get(profile.workflow)
        if prior_key is not None:
            raise ChildLaunchContractError(
                "Duplicate child launch workflow identity: "
                f"{profile.workflow} ({prior_key}, {profile.key})"
            )
        registry[profile.key] = profile
        workflows[profile.workflow] = profile.key
    return registry


CHILD_LAUNCH_PROFILE_REGISTRY = build_child_launch_profile_registry(
    PRODUCTION_CHILD_LAUNCH_PROFILES
)


def resolve_child_launch_profile(config: object) -> ChildLaunchProfile:
    """Resolve exactly one registered profile for a production runtime config."""

    key = getattr(config, "child_launch_profile", None)
    if not isinstance(key, str) or not key:
        raise TypeError(f"Unsupported child workflow config: {type(config).__name__}")
    profile = CHILD_LAUNCH_PROFILE_REGISTRY.get(key)
    if profile is None:
        raise ChildLaunchContractError(f"Unknown child launch profile: {key}")
    return profile


def child_launch_artifact_identity(
    run_dir: Path,
    artifact_path: Path,
) -> ChildLaunchArtifactIdentity:
    """Capture one no-follow identity for a parent-prepared launch artifact."""

    observed = read_contained_artifact(run_dir, artifact_path)
    return ChildLaunchArtifactIdentity(
        path=str(observed.path),
        sha256=observed.sha256,
        size_bytes=observed.size_bytes,
    )


def _validate_artifact_identity(
    run_dir: Path,
    identity: ChildLaunchArtifactIdentity,
    *,
    label: str,
) -> None:
    try:
        observed = read_contained_artifact(run_dir, identity.path)
    except (OSError, ValueError) as exc:
        raise ChildLaunchContractError(
            f"Child launch {label} artifact is unavailable: {exc}"
        ) from exc
    if observed.sha256 != identity.sha256 or observed.size_bytes != identity.size_bytes:
        raise ChildLaunchContractError(
            f"Child launch {label} artifact changed across the child launch boundary"
        )


def validate_child_launch_domain_artifacts(descriptor: ChildLaunchDescriptor) -> None:
    """Re-verify every domain-supplied artifact bound by a launch descriptor."""

    run_dir = Path(descriptor.artifacts.run_root)
    for label, identity in (
        ("capability inventory", descriptor.capability_inventory),
        ("domain policy bounds", descriptor.domain_policy_bounds),
    ):
        if identity is not None:
            _validate_artifact_identity(run_dir, identity, label=label)


def _normalized_hosts(hosts: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_host in hosts:
        host = str(raw_host).strip().lower()
        if host and host not in normalized:
            normalized.append(host)
    return tuple(normalized)


def _normalized_environment_names(names: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_name in names:
        name = str(raw_name).strip()
        if name and name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _valid_environment_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if _ENVIRONMENT_NAME.fullmatch(normalized) else None


def _codex_transport_credential_environment_names(
    config: object,
) -> tuple[str, ...]:
    """Return only credentials explicitly selected for the Codex transport."""

    explicit_api_key_env = _valid_environment_name(
        getattr(config, "codex_api_key_env", None)
    )
    if explicit_api_key_env is not None:
        return (explicit_api_key_env,)

    codex_config = getattr(config, "codex_config", None)
    if not isinstance(codex_config, Mapping):
        # An unconfigured Codex launch uses its login store. Ambient provider
        # keys are not transport authority unless the operator selects them.
        return ()
    provider_name = codex_config.get("model_provider")
    if not isinstance(provider_name, str) or not provider_name.strip():
        return ()
    provider_name = provider_name.strip()
    providers = codex_config.get("model_providers")
    provider = providers.get(provider_name) if isinstance(providers, Mapping) else None
    if isinstance(provider, Mapping):
        environment_name = _valid_environment_name(provider.get("env_key"))
        if environment_name is not None:
            return (environment_name,)
        if provider.get("requires_openai_auth") is True:
            return ("OPENAI_API_KEY",)
    # Naming the built-in OpenAI provider is the explicit opt-in available to
    # Texture and Articulation configs, which do not expose codex_api_key_env.
    if provider_name == "openai":
        return ("OPENAI_API_KEY",)
    return ()


def _configuration_flag_enabled(value: object) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _claude_sdk_uses_direct_anthropic_auth(config: object) -> bool:
    """Return whether effective Claude SDK config selects direct Anthropic auth."""

    effective_provider_flags: dict[str, object] = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("CLAUDE_CODE_USE_")
    }
    claude_config = getattr(config, "claude_config", None)
    if isinstance(claude_config, Mapping):
        configured_environment = claude_config.get("env")
        if isinstance(configured_environment, Mapping):
            effective_provider_flags.update(
                {
                    str(name): value
                    for name, value in configured_environment.items()
                    if str(name).startswith("CLAUDE_CODE_USE_")
                }
            )
    return not any(
        _configuration_flag_enabled(value)
        for value in effective_provider_flags.values()
    )


def _reasoning_only_forbidden_environment_names(
    config: object,
    *,
    runner: str,
    claude_execution_mode: str | None,
) -> tuple[str, ...]:
    """Deny domain credentials while retaining selected transport auth."""

    from world_understanding.functions.models.backends.registry import (
        load_backend_plugins,
    )
    from world_understanding.utils.credentials import (
        API_KEY_ENV_VAR_MAP,
        NIM_API_KEY_ENV_VARS,
    )

    # Credential aliases are registered by optional backend entry points. Load
    # them explicitly here so the launch policy never depends on which model
    # module happened to be imported before the reasoning child was prepared.
    load_backend_plugins()
    explicitly_forbidden = set(
        _normalized_environment_names(
            getattr(config, "child_forbidden_environment_names", ())
        )
    )
    forbidden = set(explicitly_forbidden)
    for environment_names in API_KEY_ENV_VAR_MAP.values():
        forbidden.update(_normalized_environment_names(environment_names))
    forbidden.update(NIM_API_KEY_ENV_VARS)
    forbidden.update(_REASONING_ONLY_DOMAIN_AUTHORITY_ENVIRONMENT_NAMES)

    # Preserve only an explicitly selected Codex transport credential. Default
    # launches use the login store, so ambient OPENAI_API_KEY remains denied.
    # A domain's explicit denial remains authoritative on name collisions.
    if runner == "codex":
        for environment_name in _codex_transport_credential_environment_names(config):
            if environment_name not in explicitly_forbidden:
                forbidden.discard(environment_name)

    # Direct Anthropic SDK mode may require its default credential. Alternate
    # SDK providers and Claude CLI receive no Anthropic credential exception.
    if (
        runner == "claude"
        and claude_execution_mode == "sdk"
        and _claude_sdk_uses_direct_anthropic_auth(config)
        and "ANTHROPIC_API_KEY" not in explicitly_forbidden
    ):
        forbidden.discard("ANTHROPIC_API_KEY")

    return tuple(sorted(forbidden))


def build_child_launch_descriptor(
    *,
    config: object,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    bridge_artifact_prefix: str,
    runner: str,
    model: str | None,
    model_reasoning_effort: str | None,
    claude_execution_mode: str | None,
    extra_allowed_hosts: Iterable[str] = (),
) -> ChildLaunchDescriptor:
    """Build and validate the sole launch descriptor for one child turn."""

    profile = resolve_child_launch_profile(config)
    run_root = run_dir.expanduser().resolve(strict=False)
    output_path = child_output_path.expanduser().resolve(strict=False)
    final_path = child_final_path.expanduser().resolve(strict=False)
    capability_inventory = getattr(config, "child_capability_inventory", None)
    domain_policy_bounds = getattr(config, "child_domain_policy_bounds", None)
    if capability_inventory is not None and not isinstance(
        capability_inventory, ChildLaunchArtifactIdentity
    ):
        raise ChildLaunchContractError(
            "Child launch capability inventory identity has the wrong type"
        )
    if domain_policy_bounds is not None and not isinstance(
        domain_policy_bounds, ChildLaunchArtifactIdentity
    ):
        raise ChildLaunchContractError(
            "Child launch domain policy identity has the wrong type"
        )
    if profile.requires_domain_contract and (
        capability_inventory is None or domain_policy_bounds is None
    ):
        raise ChildLaunchContractError(
            f"Child launch profile {profile.key} requires capability inventory "
            "and domain policy identities"
        )

    requested_hosts = _normalized_hosts(extra_allowed_hosts)
    if profile.network_mode == "reasoning_transport_only":
        if requested_hosts:
            raise ChildLaunchContractError(
                f"Child launch profile {profile.key} forbids domain/network hosts"
            )
        network_policy = ChildLaunchNetworkPolicy(
            mode="reasoning_transport_only",
            tool_network_access=False,
            allowed_hosts=("127.0.0.1",) if profile.scene_backend == "usd-cli" else (),
        )
        credential_mode: Literal["legacy_sanitized", "reasoning_transport_only"] = (
            "reasoning_transport_only"
        )
    else:
        legacy_hosts = list(requested_hosts)
        if profile.scene_backend == "usd-cli" and "127.0.0.1" not in legacy_hosts:
            legacy_hosts.insert(0, "127.0.0.1")
        network_policy = ChildLaunchNetworkPolicy(
            mode="legacy_runtime",
            tool_network_access=True,
            allowed_hosts=tuple(legacy_hosts),
        )
        credential_mode = "legacy_sanitized"

    if profile.network_mode == "reasoning_transport_only":
        forbidden_environment_names = _reasoning_only_forbidden_environment_names(
            config,
            runner=runner,
            claude_execution_mode=claude_execution_mode,
        )
    else:
        forbidden_environment_names = _normalized_environment_names(
            getattr(config, "child_forbidden_environment_names", ())
        )
    try:
        runner_identity = ChildLaunchRunnerIdentity(
            runner=cast(Literal["codex", "claude"], runner),
            model=model,
            model_reasoning_effort=model_reasoning_effort,
            claude_execution_mode=cast(
                Literal["sdk", "cli"] | None,
                claude_execution_mode if runner == "claude" else None,
            ),
        )
        descriptor = ChildLaunchDescriptor(
            profile_key=profile.key,
            workflow=profile.workflow,
            workflow_skill=profile.workflow_skill,
            scene_backend=profile.scene_backend,
            required_staged_skills=profile.required_staged_skills,
            skill_staging_mode=profile.skill_staging_mode,
            capability_inventory=capability_inventory,
            domain_policy_bounds=domain_policy_bounds,
            network_policy=network_policy,
            credential_policy=ChildLaunchCredentialPolicy(
                mode=credential_mode,
                forbidden_environment_names=forbidden_environment_names,
            ),
            artifacts=ChildLaunchRunContract(
                run_root=str(run_root),
                child_output_path=str(output_path),
                child_final_path=str(final_path),
                bridge_artifact_prefix=bridge_artifact_prefix,
            ),
            runner_identity=runner_identity,
        )
    except ValueError as exc:
        raise ChildLaunchContractError(
            f"Invalid child launch descriptor: {exc}"
        ) from exc

    if (
        descriptor.workflow != profile.workflow
        or descriptor.workflow_skill != profile.workflow_skill
        or descriptor.required_staged_skills != profile.required_staged_skills
    ):
        raise ChildLaunchContractError(
            f"Child launch descriptor does not match profile {profile.key}"
        )
    validate_child_launch_domain_artifacts(descriptor)
    return descriptor


def descriptor_request_payload(
    descriptor: ChildLaunchDescriptor,
) -> Mapping[str, Any]:
    """Return the immutable provider-request projection of one descriptor."""

    return descriptor.model_dump(mode="json")
