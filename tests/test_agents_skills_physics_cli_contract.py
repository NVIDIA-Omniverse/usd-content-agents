# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the public Physics Agent CLI skill."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "physics-agent-cli" / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _section(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def test_cli_run_stays_in_foreground_until_a_terminal_outcome() -> None:
    text = _normalized(_skill_text())

    required = (
        "Wait for the command to reach a terminal outcome before returning.",
        "Treat every required `physics-agent` invocation",
        "`run`, `predict`, `tune`, `refine`, and dataset commands",
        "as foreground work",
        "one-shot or headless agent run",
        "Do not append `&`, use `nohup`",
        "create an ordinary detached shell job",
        "Session teardown can kill that process",
    )
    for snippet in required:
        assert snippet in text, f"missing foreground contract: {snippet!r}"


def test_cli_does_not_invent_a_wakeup_or_shell_job_handoff() -> None:
    text = _normalized(_skill_text())

    required = (
        "Do not promise to resume, poll, or wake up later",
        "cannot fire after the current agent run ends",
        "durable external monitor or handoff",
        "survives session teardown",
        "job or monitor ID, owner, log location, and exact status command",
        "A local PID or shell job is not a durable handoff.",
    )
    for snippet in required:
        assert snippet in text, f"missing durable-handoff contract: {snippet!r}"


def test_cli_routes_each_anti_trigger_to_the_exact_sibling_skill() -> None:
    text = _normalized(_skill_text())

    routes = (
        "Call an already-running Physics Agent REST API, monitor its sessions, "
        "or download service artifacts | `physics-agent-client`",
        "Build, start, stop, or configure the local Physics Agent Docker "
        "Compose service | `deploy-physics-agent-docker`",
        "Provision Brev-hosted render/VLM dependencies or run the Brev hybrid "
        "Physics Agent workflow | `deploy-physics-agent-brev`",
        "Assign or refine visual/PBR materials rather than physical properties "
        "| `material-agent-cli`",
    )
    for route in routes:
        assert route in text, f"missing sibling-skill route: {route!r}"

    assert "Use service or Docker deploy skills instead" not in text


def test_cli_keeps_rest_and_deployment_commands_outside_its_boundary() -> None:
    text = _normalized(_skill_text())

    assert "This skill runs the local CLI." in text
    assert (
        "Do not construct REST requests or manage service containers from it." in text
    )
    assert (
        "`physics-agent-client` for `/pipeline`, `/predict`, `/tune`, and `/refine`"
        in text
    )
    assert "curl -X POST" not in text
    assert '"$BASE_URL/pipeline"' not in text


def test_cli_reports_common_and_foreground_terminal_fields() -> None:
    text = _skill_text()
    common = _normalized(
        _section(text, "## Output Format", "### Foreground Terminal Run")
    )
    foreground = _normalized(
        _section(text, "### Foreground Terminal Run", "### Durable External Handoff")
    )

    common_fields = (
        "Report these items for every execution path:",
        "Config path and session ID, when created.",
        "Working directory",
        "Key artifacts when present:",
    )
    for snippet in common_fields:
        assert snippet in common, f"missing common reporting contract: {snippet!r}"

    assert "Physics Agent CLI terminal outcome" in foreground
    assert "`completed`, `failed`, or `interrupted`" in foreground
    assert "and its exit code" in foreground
    assert (
        "A started process or partial log is not evidence of completion." in foreground
    )
    assert "handoff-creation command" not in foreground


def test_cli_reports_durable_handoff_creation_and_external_status_separately() -> None:
    handoff = _normalized(
        _section(_skill_text(), "### Durable External Handoff", "## Troubleshooting")
    )

    assert "handoff-creation command's terminal outcome and exit code" in handoff
    assert "only whether handoff creation succeeded" in handoff
    assert "not the external Physics Agent job's eventual outcome" in handoff
    assert (
        "external job ID and, when distinct, the monitor ID, plus the owner, log "
        "location, exact status command" in handoff
    )
    assert "currently observed external job status" in handoff
    assert "Label that status as non-terminal" in handoff


def test_cli_preserves_the_supported_linux_and_wsl2_runtime_policy() -> None:
    text = _normalized(_skill_text())

    assert "Requires Linux, a Linux container, or WSL2" in text
    assert "Official runtime targets are Linux, Linux containers, and WSL2." in text
    assert "Native Windows shell execution is not supported" in text
    assert "direct the user to WSL2 or Linux" in text
    assert "Windows PowerShell:" not in text
    assert r".\.venv\Scripts\Activate.ps1" not in text


def test_cli_skill_version_tracks_the_contract_revision() -> None:
    assert 'version: "0.1.2"' in _skill_text()
