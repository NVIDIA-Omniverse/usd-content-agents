# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard the reviewed public fixed-pipeline SkillSpector baseline."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = REPO_ROOT / ".agents" / "skills" / "fixed-pipeline"
BASELINE_PATH = SKILL_ROOT / "config" / "skillspector-baseline.yaml"
ENVIRONMENT_FILE_PATHS = {
    "references/deploy-collection/reference.md",
    "references/deploy-embeddings-brev/reference.md",
    "references/deploy-image-gen-brev/reference.md",
    "references/deploy-material-agent-brev/reference.md",
    "references/deploy-material-agent-docker/reference.md",
    "references/deploy-physics-agent-brev/reference.md",
    "references/deploy-physics-agent-docker/reference.md",
    "references/deploy-qwen-vlm-brev/reference.md",
    "references/deploy-qwen-vlm-brev/references/qwen36-h100.md",
    "references/deploy-texture-agent-brev/reference.md",
    "references/deploy-texture-agent-docker/reference.md",
}
SSH_HOST_REGISTRY_PATH = "references/deploy-qwen-vlm-brev/reference.md"
DOCKER_IMAGE_PATHS = {
    "references/deploy-image-gen-brev/reference.md",
    "references/deploy-material-agent-brev/reference.md",
    "references/deploy-material-agent-docker/reference.md",
    "references/deploy-ovrtx-docker/reference.md",
    "references/deploy-physics-agent-brev/reference.md",
    "references/deploy-physics-agent-docker/reference.md",
    "references/deploy-qwen-vlm-brev/reference.md",
    "references/deploy-qwen-vlm-brev/references/qwen36-h100.md",
    "references/deploy-texture-agent-brev/reference.md",
    "references/deploy-texture-agent-docker/reference.md",
}
DECLARED_PERMISSIONS_PATH = "SKILL.md"
ANTI_REFUSAL_PATHS = {
    "references/material-agent-cli/evals/fixtures/prefailed-resume/dataset/dataset.json",
    "references/material-agent-cli/references/config-template.yaml",
}
SESSION_CLEANUP_PATHS = {
    "references/material-agent-client/evals/evals.json",
    "references/physics-agent-client/evals/evals.json",
}


def test_fixed_pipeline_skillspector_baseline_is_narrow_and_reviewed() -> None:
    baseline_text = BASELINE_PATH.read_text(encoding="utf-8")
    baseline = yaml.safe_load(baseline_text)

    assert baseline["version"] == 2
    assert baseline["fingerprints"] == []
    rules = baseline["rules"]
    assert len(rules) == (
        len(ENVIRONMENT_FILE_PATHS)
        + len(DOCKER_IMAGE_PATHS)
        + len(SESSION_CLEANUP_PATHS)
        + 4
    )

    environment_rules = {
        rule["path"] for rule in rules if rule.get("message") == "*.[e]nv*"
    }
    assert environment_rules == ENVIRONMENT_FILE_PATHS
    assert {rule["id"] for rule in rules if rule.get("message") == "*.[e]nv*"} == {
        "PE3"
    }
    assert {
        rule["path"] for rule in rules if rule.get("message") == "*known[_]hosts*"
    } == {SSH_HOST_REGISTRY_PATH}
    assert {
        rule["id"] for rule in rules if rule.get("message") == "*known[_]hosts*"
    } == {"PE3"}
    assert {rule["path"] for rule in rules if rule["id"] == "RP1"} == DOCKER_IMAGE_PATHS
    assert {rule["path"] for rule in rules if rule["id"] == "LP3"} == {
        DECLARED_PERMISSIONS_PATH
    }
    assert {
        rule["path"] for rule in rules if rule.get("message") == "D[O] NOT judge"
    } == ANTI_REFUSAL_PATHS
    assert {
        rule["id"] for rule in rules if rule.get("message") == "D[O] NOT judge"
    } == {"AR2"}
    assert {
        rule["path"]
        for rule in rules
        if rule.get("message")
        in {"DE[L]ETE /sessions/{id}", "DE[L]ETE /sessions/{id}."}
    } == SESSION_CLEANUP_PATHS
    assert {
        rule["id"]
        for rule in rules
        if rule.get("message")
        in {"DE[L]ETE /sessions/{id}", "DE[L]ETE /sessions/{id}."}
    } == {"TM1"}

    for rule in rules:
        if rule["id"] in {"LP3", "RP1"}:
            assert "message" not in rule

    for rule in rules:
        assert rule["id"] in {"AR2", "LP3", "PE3", "RP1", "TM1"}
        assert "*" not in rule["path"]
        assert (SKILL_ROOT / rule["path"]).is_file()
        assert rule["reason"].startswith("Reviewed ")
    assert ".env" not in baseline_text
    assert "known_hosts" not in baseline_text
    assert "DO NOT judge" not in baseline_text
    assert "DELETE /sessions" not in baseline_text
    assert "read private keys" not in baseline_text
