# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Physics Agent refine-loop skill coverage."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"


def _read_skill(name: str) -> str:
    return (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


def _assert_contains(text: str, snippet: str, *, source: str) -> None:
    assert snippet in text, f"{source} missing required snippet: {snippet!r}"


def test_physics_client_skill_documents_refine_service_flow() -> None:
    text = _read_skill("physics-agent-client")

    required_snippets = (
        "Use `/refine`",
        "POST /refine",
        "GET /refine/{id}/status",
        "source_session_id",
        "scenario_yaml",
        "optimizer=botorch",
        "score_threshold=0.9",
        "seed=42",
        "`PA_REFINE_BACKEND` / `PA_REFINE_MODEL`",
        "final/tuned_physics.usda",
        "| `max_iterations` | No | `5` |",
    )
    for snippet in required_snippets:
        _assert_contains(text, snippet, source="physics-agent-client")


def test_deploy_skills_document_provider_neutral_refine_runtime() -> None:
    docker_text = _read_skill("deploy-physics-agent-docker")
    brev_text = _read_skill("deploy-physics-agent-brev")

    _assert_contains(
        docker_text,
        "`PA_REFINE_BACKEND` and `PA_REFINE_MODEL`",
        source="deploy-physics-agent-docker",
    )
    _assert_contains(
        docker_text,
        "registered chat/VLM provider",
        source="deploy-physics-agent-docker",
    )
    _assert_contains(
        docker_text,
        "provider's credential",
        source="deploy-physics-agent-docker",
    )
    _assert_contains(
        brev_text,
        "registered chat/VLM provider",
        source="deploy-physics-agent-brev",
    )
    _assert_contains(
        brev_text,
        "provider's credential",
        source="deploy-physics-agent-brev",
    )


def test_public_physics_refine_docs_do_not_claim_internal_only_backends() -> None:
    paths = (
        SKILLS_DIR / "deploy-physics-agent-docker" / "SKILL.md",
        SKILLS_DIR / "deploy-physics-agent-brev" / "SKILL.md",
        SKILLS_DIR / "deploy-physics-agent-brev" / "agents" / "openai.yaml",
        SKILLS_DIR / "physics-agent-client" / "SKILL.md",
        SKILLS_DIR / "physics-agent-client" / "agents" / "openai.yaml",
        REPO_ROOT / "apps" / "physics_agent_service" / "client" / "README.md",
        REPO_ROOT / "apps" / "physics_agent_service" / "docs" / "api.md",
        REPO_ROOT / "apps" / "physics_agent" / "docs" / "tuning.md",
    )
    stale_phrases = (
        "internal-only refine",
        "internal-only model configuration",
        "internal service refine",
        "internal refine",
        "internal iterative",
        "internal `/refine`",
        "internal nvidia deployments",
        "rejects public backend choices",
        "internal backend config",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in stale_phrases:
            assert phrase not in text, f"{path} contains stale wording: {phrase!r}"


def test_physics_cli_metadata_is_refine_discoverable() -> None:
    skill_text = _read_skill("physics-agent-cli")
    agent_text = (
        SKILLS_DIR / "physics-agent-cli" / "agents" / "openai.yaml"
    ).read_text(encoding="utf-8")

    _assert_contains(
        skill_text,
        "refine-loop workflows",
        source="physics-agent-cli/SKILL.md",
    )
    _assert_contains(
        agent_text,
        "iterative refine",
        source="physics-agent-cli/agents/openai.yaml",
    )
