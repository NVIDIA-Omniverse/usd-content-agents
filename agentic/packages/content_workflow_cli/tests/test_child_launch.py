# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for every production shared child-launch consumer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Protocol

import pytest
from pydantic import ValidationError

from content_workflow_cli.articulation_runner import (
    _ArticulationChildRuntimeConfig,
)
from content_workflow_cli.asset_runner import AssetRunConfig
from content_workflow_cli.child_launch import (
    ARTICULATION_CHILD_LAUNCH_PROFILE,
    CHILD_LAUNCH_PROFILE_REGISTRY,
    TEXTURE_CHILD_LAUNCH_PROFILE,
    VALIDATION_CHILD_LAUNCH_PROFILE,
    ChildLaunchContractError,
    ChildLaunchNetworkPolicy,
    ChildLaunchProfile,
    build_child_launch_descriptor,
    build_child_launch_profile_registry,
    child_launch_artifact_identity,
    resolve_child_launch_profile,
    validate_child_launch_domain_artifacts,
)
from content_workflow_cli.external_refine_runner import PhysicsExternalRefineConfig
from content_workflow_cli.material_authoring_runner import (
    _MaterialAuthoringChildConfig,
)
from content_workflow_cli.mesh_segmentation_runner import MeshSegmentationConfig
from content_workflow_cli.runner import (
    MaterialAssignConfig,
    PhysicsApplyConfig,
    _child_agent_environment,
    _child_workflow_name,
    _stage_agent_skills,
)
from content_workflow_cli.scene_runner import SceneRunConfig
from content_workflow_cli.texture_runner import _TextureChildRuntimeConfig
from content_workflow_cli.validation_runner import _ValidationChildConfig

PRODUCTION_SHARED_LAUNCH_CONFIGS = (
    (AssetRunConfig, "asset.run"),
    (MaterialAssignConfig, "materials.assign"),
    (PhysicsApplyConfig, "physics.apply"),
    (SceneRunConfig, "scene.run"),
    (MeshSegmentationConfig, "mesh-segmentation.run"),
    (_MaterialAuthoringChildConfig, "materials.author"),
    (PhysicsExternalRefineConfig, "physics.refine_external.agentic"),
    (_ArticulationChildRuntimeConfig, ARTICULATION_CHILD_LAUNCH_PROFILE),
    (_TextureChildRuntimeConfig, TEXTURE_CHILD_LAUNCH_PROFILE),
    (_ValidationChildConfig, VALIDATION_CHILD_LAUNCH_PROFILE),
)


class _LaunchConfigType(Protocol):
    child_launch_profile: ClassVar[str]


@pytest.mark.parametrize(
    ("config_type", "profile_key"), PRODUCTION_SHARED_LAUNCH_CONFIGS
)
def test_every_production_shared_launch_config_has_one_registered_descriptor(
    config_type: type[_LaunchConfigType],
    profile_key: str,
) -> None:
    profile = resolve_child_launch_profile(config_type)

    assert config_type.child_launch_profile == profile_key
    assert profile.key == profile_key
    assert _child_workflow_name(config_type) == profile.workflow
    assert profile.workflow_skill in profile.required_staged_skills


def test_network_policy_accepts_bracketed_ipv6_hosts() -> None:
    policy = ChildLaunchNetworkPolicy(
        mode="legacy_runtime",
        tool_network_access=True,
        allowed_hosts=("[::1]",),
    )

    assert policy.allowed_hosts == ("[::1]",)


def test_network_policy_rejects_bracketed_non_ipv6_hosts() -> None:
    with pytest.raises(ValidationError, match="invalid child launch network host"):
        ChildLaunchNetworkPolicy(
            mode="legacy_runtime",
            tool_network_access=True,
            allowed_hosts=("[provider.example]",),
        )


def test_reasoning_network_policy_allows_only_scoped_loopback_adapters() -> None:
    policy = ChildLaunchNetworkPolicy(
        mode="reasoning_transport_only",
        tool_network_access=False,
        allowed_hosts=("127.0.0.1",),
    )

    assert policy.allowed_hosts == ("127.0.0.1",)
    with pytest.raises(ValidationError, match="only loopback adapters"):
        ChildLaunchNetworkPolicy(
            mode="reasoning_transport_only",
            tool_network_access=False,
            allowed_hosts=("provider.example",),
        )


def test_production_shared_launch_profile_identities_are_exhaustive() -> None:
    expected = {profile_key for _, profile_key in PRODUCTION_SHARED_LAUNCH_CONFIGS}

    assert set(CHILD_LAUNCH_PROFILE_REGISTRY) == expected
    assert len(expected) == len(PRODUCTION_SHARED_LAUNCH_CONFIGS)


def test_unknown_and_unregistered_configs_fail_closed() -> None:
    with pytest.raises(TypeError, match="Unsupported child workflow config"):
        resolve_child_launch_profile(object())

    unknown = type("UnknownConfig", (), {"child_launch_profile": "unknown.run"})
    with pytest.raises(ChildLaunchContractError, match="Unknown child launch profile"):
        resolve_child_launch_profile(unknown)


def test_duplicate_profile_keys_and_workflows_fail_closed() -> None:
    profile = CHILD_LAUNCH_PROFILE_REGISTRY["asset.run"]

    with pytest.raises(ChildLaunchContractError, match="Duplicate.*profile key"):
        build_child_launch_profile_registry((profile, profile))
    with pytest.raises(ChildLaunchContractError, match="Duplicate.*workflow identity"):
        build_child_launch_profile_registry(
            (profile, profile.model_copy(update={"key": "another.asset.run"}))
        )


def test_incomplete_profile_fails_validation() -> None:
    with pytest.raises(ValidationError, match="require at least one staged skill"):
        ChildLaunchProfile(
            key="incomplete.run",
            workflow="incomplete.run",
            workflow_skill="missing-skill",
            scene_backend="usd-cli",
            required_staged_skills=(),
        )


@pytest.mark.parametrize(
    "profile_key",
    (ARTICULATION_CHILD_LAUNCH_PROFILE, TEXTURE_CHILD_LAUNCH_PROFILE),
)
def test_domain_launch_requires_frozen_inventory_and_policy(
    tmp_path: Path,
    profile_key: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = type("DomainConfig", (), {"child_launch_profile": profile_key})()

    with pytest.raises(ChildLaunchContractError, match="requires capability inventory"):
        build_child_launch_descriptor(
            config=config,
            run_dir=run_dir,
            child_output_path=run_dir / "raw" / "output.jsonl",
            child_final_path=run_dir / "raw" / "final.json",
            bridge_artifact_prefix="domain",
            runner="codex",
            model=None,
            model_reasoning_effort=None,
            claude_execution_mode=None,
        )


@pytest.mark.parametrize(
    "profile_key",
    (ARTICULATION_CHILD_LAUNCH_PROFILE, TEXTURE_CHILD_LAUNCH_PROFILE),
)
def test_domain_launch_rejects_any_tool_network_host(
    tmp_path: Path,
    profile_key: str,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": profile_key,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
        },
    )()

    with pytest.raises(ChildLaunchContractError, match="forbids domain/network hosts"):
        build_child_launch_descriptor(
            config=config,
            run_dir=run_dir,
            child_output_path=run_dir / "raw" / "output.jsonl",
            child_final_path=run_dir / "raw" / "final.json",
            bridge_artifact_prefix="domain",
            runner="codex",
            model=None,
            model_reasoning_effort=None,
            claude_execution_mode=None,
            extra_allowed_hosts=("provider.example",),
        )


@pytest.mark.parametrize(
    ("runner", "claude_execution_mode", "preserved_environment_names"),
    (
        ("codex", None, set()),
        ("claude", "sdk", {"ANTHROPIC_API_KEY"}),
        ("claude", "cli", set()),
    ),
)
def test_every_reasoning_only_profile_denies_parent_provider_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    claude_execution_mode: str | None,
    preserved_environment_names: set[str],
) -> None:
    for environment_name in tuple(os.environ):
        if environment_name.startswith("CLAUDE_CODE_USE_"):
            monkeypatch.delenv(environment_name)
    provider_environment_names = {
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "NGC_API_KEY",
        "NVIDIA_API_KEY",
        "NVCF_API_KEY",
        "NVCF_RENDER_FUNCTION_ID",
        "OPENAI_API_KEY",
        "MA_NIM_API_KEY",
        "PA_NIM_API_KEY",
        "TA_NIM_API_KEY",
        "WU_NIM_API_KEY",
    }
    for environment_name in provider_environment_names:
        monkeypatch.setenv(environment_name, f"dummy-{environment_name.lower()}")

    for profile in CHILD_LAUNCH_PROFILE_REGISTRY.values():
        if profile.network_mode != "reasoning_transport_only":
            continue
        run_dir = tmp_path / profile.key
        run_dir.mkdir()
        artifact = run_dir / "domain.json"
        artifact.write_text("{}\n", encoding="utf-8")
        identity = child_launch_artifact_identity(run_dir, artifact)
        config = type(
            "DomainConfig",
            (),
            {
                "child_launch_profile": profile.key,
                "child_capability_inventory": identity,
                "child_domain_policy_bounds": identity,
                "child_forbidden_environment_names": ("DOMAIN_ONLY_TOKEN",),
            },
        )()
        descriptor = build_child_launch_descriptor(
            config=config,
            run_dir=run_dir,
            child_output_path=run_dir / "raw" / "output.jsonl",
            child_final_path=run_dir / "raw" / "final.json",
            bridge_artifact_prefix="domain",
            runner=runner,
            model=None,
            model_reasoning_effort=None,
            claude_execution_mode=claude_execution_mode,
        )

        forbidden = set(descriptor.credential_policy.forbidden_environment_names)
        assert "DOMAIN_ONLY_TOKEN" in forbidden
        assert provider_environment_names - preserved_environment_names <= forbidden
        assert preserved_environment_names.isdisjoint(forbidden)
        child_environment = _child_agent_environment(descriptor)
        assert forbidden.isdisjoint(child_environment)
        assert all(
            environment_name in child_environment
            for environment_name in preserved_environment_names
        )


@pytest.mark.parametrize(
    ("runner", "claude_execution_mode", "environment_name"),
    (
        ("codex", None, "OPENAI_API_KEY"),
        ("claude", "sdk", "ANTHROPIC_API_KEY"),
    ),
)
def test_reasoning_only_explicit_domain_denial_overrides_default_transport_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    claude_execution_mode: str | None,
    environment_name: str,
) -> None:
    monkeypatch.setenv(environment_name, "dummy-domain-credential")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": TEXTURE_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "child_forbidden_environment_names": (environment_name,),
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner=runner,
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=claude_execution_mode,
    )

    assert environment_name in descriptor.credential_policy.forbidden_environment_names
    assert environment_name not in _child_agent_environment(descriptor)


def test_reasoning_only_codex_launch_preserves_explicit_transport_credential(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": TEXTURE_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "codex_api_key_env": "NVIDIA_API_KEY",
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    forbidden = set(descriptor.credential_policy.forbidden_environment_names)
    assert "NVIDIA_API_KEY" not in forbidden
    assert "OPENAI_API_KEY" in forbidden
    assert "NGC_API_KEY" in forbidden


def test_reasoning_only_codex_launch_preserves_selected_provider_env_key(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": ARTICULATION_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "codex_config": {
                "model_provider": "nvidia-proxy",
                "model_providers": {"nvidia-proxy": {"env_key": "NVIDIA_API_KEY"}},
            },
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    forbidden = set(descriptor.credential_policy.forbidden_environment_names)
    assert "NVIDIA_API_KEY" not in forbidden
    assert "OPENAI_API_KEY" in forbidden
    assert "GOOGLE_API_KEY" in forbidden


@pytest.mark.parametrize(
    "codex_config",
    (
        {"model_provider": "openai"},
        {
            "model_provider": "openai-proxy",
            "model_providers": {"openai-proxy": {"requires_openai_auth": True}},
        },
    ),
)
def test_reasoning_only_codex_launch_preserves_explicit_openai_transport(
    tmp_path: Path,
    codex_config: dict[str, object],
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": TEXTURE_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "codex_config": codex_config,
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    assert (
        "OPENAI_API_KEY" not in descriptor.credential_policy.forbidden_environment_names
    )


@pytest.mark.parametrize(
    "transport_config",
    (
        {"codex_api_key_env": "NVIDIA_API_KEY"},
        {
            "codex_config": {
                "model_provider": "nvidia-proxy",
                "model_providers": {"nvidia-proxy": {"env_key": "NVIDIA_API_KEY"}},
            }
        },
    ),
)
def test_reasoning_only_codex_transport_cannot_override_domain_denial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport_config: dict[str, object],
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "dummy-domain-credential")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": TEXTURE_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "child_forbidden_environment_names": ("NVIDIA_API_KEY",),
            **transport_config,
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    assert "NVIDIA_API_KEY" in descriptor.credential_policy.forbidden_environment_names
    assert "NVIDIA_API_KEY" not in _child_agent_environment(descriptor)


@pytest.mark.parametrize("selector_source", ("config", "ambient"))
def test_reasoning_only_alternate_claude_provider_denies_anthropic_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector_source: str,
) -> None:
    for environment_name in tuple(os.environ):
        if environment_name.startswith("CLAUDE_CODE_USE_"):
            monkeypatch.delenv(environment_name)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-domain-credential")
    claude_config: dict[str, object] | None = None
    if selector_source == "config":
        claude_config = {"env": {"CLAUDE_CODE_USE_BEDROCK": "1"}}
    else:
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": ARTICULATION_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "claude_config": claude_config,
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="claude",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode="sdk",
    )

    assert (
        "ANTHROPIC_API_KEY" in descriptor.credential_policy.forbidden_environment_names
    )
    assert "ANTHROPIC_API_KEY" not in _child_agent_environment(descriptor)


def test_reasoning_only_loads_backend_aliases_before_freezing_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from world_understanding.functions.models.backends import registry
    from world_understanding.utils.credentials import API_KEY_ENV_VAR_MAP

    def register_late_backend() -> tuple[str, ...]:
        monkeypatch.setitem(
            API_KEY_ENV_VAR_MAP,
            "late_provider",
            ("LATE_PROVIDER_API_KEY",),
        )
        return ("late-provider:test",)

    monkeypatch.setattr(registry, "load_backend_plugins", register_late_backend)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": TEXTURE_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
        },
    )()

    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    assert (
        "LATE_PROVIDER_API_KEY"
        in descriptor.credential_policy.forbidden_environment_names
    )


@pytest.mark.parametrize(
    "profile_key",
    (ARTICULATION_CHILD_LAUNCH_PROFILE, TEXTURE_CHILD_LAUNCH_PROFILE),
)
def test_domain_launch_stages_only_descriptor_required_skills(
    tmp_path: Path,
    profile_key: str,
) -> None:
    profile = CHILD_LAUNCH_PROFILE_REGISTRY[profile_key]
    repo_root = tmp_path / "repo"
    source_skills = repo_root / "agentic" / ".agents" / "skills"
    for skill in (*profile.required_staged_skills, "unrelated-skill"):
        skill_dir = source_skills / skill
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": profile_key,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
            "repo_root": repo_root,
        },
    )()
    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )

    _stage_agent_skills(config, run_dir, launch_descriptor=descriptor)

    for discovery_root in (run_dir / ".agents", run_dir / ".claude"):
        assert {path.name for path in (discovery_root / "skills").iterdir()} == set(
            profile.required_staged_skills
        )


def test_domain_artifact_identity_drift_fails_closed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "domain.json"
    artifact.write_text("{}\n", encoding="utf-8")
    identity = child_launch_artifact_identity(run_dir, artifact)
    config = type(
        "DomainConfig",
        (),
        {
            "child_launch_profile": ARTICULATION_CHILD_LAUNCH_PROFILE,
            "child_capability_inventory": identity,
            "child_domain_policy_bounds": identity,
        },
    )()
    descriptor = build_child_launch_descriptor(
        config=config,
        run_dir=run_dir,
        child_output_path=run_dir / "raw" / "output.jsonl",
        child_final_path=run_dir / "raw" / "final.json",
        bridge_artifact_prefix="domain",
        runner="codex",
        model=None,
        model_reasoning_effort=None,
        claude_execution_mode=None,
    )
    artifact.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(
        ChildLaunchContractError, match="changed across the child launch"
    ):
        validate_child_launch_domain_artifacts(descriptor)
