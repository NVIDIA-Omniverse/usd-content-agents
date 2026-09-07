# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for credit-safe Brev agent-service deployment planning."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "brev_agent_services.py"


def _load_planner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("brev_agent_services", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BREV = _load_planner()
POSIX_SHELL_ONLY = pytest.mark.skipif(
    sys.platform == "win32",
    reason="credential-transfer execution tests require Bash and POSIX tools",
)


def _plan(
    service: str,
    preset: str,
    **kwargs: object,
) -> object:
    return BREV.build_plan(
        BREV.PlannerOptions(
            service=service,
            preset=preset,
            name="wu-test",
            **kwargs,
        )
    )


def test_material_hybrid_uses_rtx_render_and_a100_qwen_vlm_defaults() -> None:
    plan = _plan("material", "hybrid")

    render_node, vlm_node = plan.nodes

    assert "--gpu-name" in render_node.dry_run.argv
    assert "RTX" in render_node.dry_run.argv
    assert "A100" not in render_node.dry_run.argv
    assert "--stoppable" in render_node.dry_run.argv

    assert "--gpu-name" in vlm_node.dry_run.argv
    assert "A100" in vlm_node.dry_run.argv
    assert "--min-vram" in vlm_node.dry_run.argv
    assert "80" in vlm_node.dry_run.argv

    assert "MA_VLM_MODEL=Qwen/Qwen3.5-4B" in plan.environment
    assert "MA_LLM_MODEL=Qwen/Qwen3.5-4B" in plan.environment
    assert "RENDER_ENDPOINT=http://localhost:8001" in plan.environment
    assert "MA_RENDERING_USE_DATA_URI=true" in plan.environment
    assert "MA_VLM_NIM_BASE_URL=http://localhost:8003/v1" in plan.environment
    assert "MA_NIM_API_KEY=not-used" in plan.environment
    assert plan.environment_description.startswith("Set these on the local machine")
    assert any(
        "does not require Brev instance-to-instance" in note for note in plan.notes
    )

    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)
    assert "ovrtx-rendering-api" in deploy_shell
    assert "rsync -az --delete" in deploy_shell
    assert "--exclude .build-resources" in deploy_shell
    assert "--exclude .env" in deploy_shell
    assert "--exclude .envrc" in deploy_shell
    assert "--exclude '*credentials*.json'" in deploy_shell
    assert "--exclude '*private*key*'" in deploy_shell
    assert "--exclude '*.key'" in deploy_shell
    assert "/home/ubuntu/world-understanding" in deploy_shell
    assert "apps/ovrtx_rendering_api/docker-compose.yml" in deploy_shell
    assert "apps/physics_agent_service/docker-compose.yml" not in deploy_shell
    assert "OVRTX_RENDER_MODE=pt" in deploy_shell
    assert "OVRTX_HOST_PORT=8001" in deploy_shell
    assert "Docker Root Dir" in deploy_shell
    assert "df -h / /home /mnt/*" in deploy_shell
    assert "mkdir -p /home/ubuntu/world-understanding/.build-resources" in deploy_shell
    assert deploy_shell.index("--exclude .build-resources") < deploy_shell.index(
        "mkdir -p /home/ubuntu/world-understanding/.build-resources"
    )
    assert deploy_shell.index(
        "mkdir -p /home/ubuntu/world-understanding/.build-resources"
    ) < deploy_shell.index(
        "docker compose -f apps/ovrtx_rendering_api/docker-compose.yml"
    )
    assert "brev port-forward wu-test-material-render -p 8001:8001" in deploy_shell
    assert "brev port-forward wu-test-material-vlm -p 8003:8000" in deploy_shell


def test_ovrtx_dockerfile_still_copies_build_resources_path() -> None:
    dockerfile = (REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "COPY .build-resources /app/.build-resources" in dockerfile


def test_rendered_endpoint_wiring_exports_assignment_lines() -> None:
    plan = _plan("physics", "hybrid")
    markdown = plan.render_markdown()

    assert "export RENDER_ENDPOINT=http://localhost:8001" in markdown
    assert "export MA_RENDERING_USE_DATA_URI=true" in markdown
    assert "export PA_VLM_NIM_BASE_URL=http://localhost:8003/v1" in markdown
    assert "export PA_NIM_API_KEY=not-used" in markdown
    assert "\nRENDER_ENDPOINT=http://localhost:8001\n" not in markdown


def test_material_hybrid_can_target_external_model_base_url() -> None:
    plan = _plan(
        "material",
        "hybrid",
        hybrid_connectivity="external-url",
        model_base_url="https://vlm.example.test/v1",
    )

    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert "MA_VLM_NIM_BASE_URL=https://vlm.example.test/v1" in plan.environment
    assert "MA_LLM_NIM_BASE_URL=https://vlm.example.test/v1" in plan.environment
    assert "MA_NIM_API_KEY=not-used" not in plan.environment
    assert any("MA_NIM_API_KEY" in entry for entry in plan.environment)
    assert "printf" in deploy_shell
    assert "MA_VLM_NIM_BASE_URL=https://vlm.example.test/v1" in deploy_shell
    assert "MA_LLM_NIM_BASE_URL=https://vlm.example.test/v1" in deploy_shell
    assert "> .env" in deploy_shell
    assert (
        "docker compose --env-file .env -f apps/material_agent_service/docker-compose.yml"
        in deploy_shell
    )
    assert "https://vlm.example.test/v1/models" in deploy_shell


def test_physics_private_ip_hybrid_does_not_emit_placeholder_key() -> None:
    plan = _plan("physics", "hybrid", hybrid_connectivity="private-ip")

    assert "PA_NIM_API_KEY=not-used" not in plan.environment
    assert any("PA_NIM_API_KEY" in entry for entry in plan.environment)


def test_non_local_hybrid_placeholder_urls_are_shell_safe() -> None:
    for connectivity in ("external-url", "tunnel-url", "private-ip"):
        plan = _plan("material", "hybrid", hybrid_connectivity=connectivity)
        env_block = "\n".join(plan.environment)

        assert "<" not in env_block
        assert ">" not in env_block
        assert "replace-with-" in env_block


def test_render_only_hosted_provider_uses_service_model_defaults() -> None:
    plan = _plan("material", "render-only")

    assert "MA_VLM_MODEL=Qwen/Qwen3.5-4B" not in plan.environment
    assert "MA_LLM_MODEL=Qwen/Qwen3.5-4B" not in plan.environment
    assert any("service defaults" in entry for entry in plan.environment)


def test_render_only_can_override_hosted_provider_model() -> None:
    plan = _plan("physics", "render-only", qwen_model="qwen/hosted-model")

    assert "PA_VLM_MODEL=qwen/hosted-model" in plan.environment


def test_same_node_hybrid_is_rejected_for_render_required_services() -> None:
    with pytest.raises(ValueError, match="single-host-local-vlm"):
        _plan("material", "hybrid", hybrid_connectivity="same-node")


def test_h100_is_rejected_for_render_node_presets() -> None:
    with pytest.raises(ValueError, match="not valid for an OVRTX render node"):
        _plan("physics", "render-only", render_gpu_name="H100")


def test_physics_single_host_local_vlm_uses_existing_overlay() -> None:
    plan = _plan("physics", "single-host-local-vlm")
    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert "apps/physics_agent_service/docker-compose.yml" in deploy_shell
    assert "apps/physics_agent_service/docker-compose.multi-gpu.yml" in deploy_shell
    assert "--profile vlm" in deploy_shell
    assert "PA_VLM_MODEL=nvidia/cosmos-reason2-8b" in plan.environment
    assert "PA_VLM_NIM_BASE_URL=http://vlm-nim:8000/v1" in plan.environment


def test_physics_local_pipeline_hybrid_uses_render_data_uri_flag() -> None:
    plan = _plan("physics", "hybrid")

    assert "MA_RENDERING_USE_DATA_URI=true" in plan.environment
    assert "PA_VLM_NIM_BASE_URL=http://localhost:8003/v1" in plan.environment


def test_texture_service_only_uses_cpu_type_and_no_gpu_filter() -> None:
    plan = _plan("texture", "service-only")
    node = plan.nodes[0]

    assert node.dry_run.argv == (
        "brev",
        "create",
        "wu-test-texture-service",
        "--dry-run",
        "--type",
        "n2d-standard-4",
    )
    assert "--gpu-name" not in node.dry_run.argv
    assert "TA_LLM_MODEL=Qwen/Qwen3.5-4B" not in plan.environment
    assert any("service defaults" in entry for entry in plan.environment)


def test_texture_service_only_can_override_hosted_llm_model() -> None:
    plan = _plan("texture", "service-only", qwen_model="qwen/hosted-model")

    assert "TA_LLM_MODEL=qwen/hosted-model" in plan.environment


def test_texture_hybrid_defaults_to_brev_image_gen_without_llm() -> None:
    plan = _plan("texture", "hybrid")
    (image_gen_node,) = plan.nodes
    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert image_gen_node.dry_run.argv == (
        "brev",
        "create",
        "wu-test-texture-image-gen",
        "--dry-run",
        "--type",
        "g6e.xlarge",
        "--min-disk",
        "500",
    )
    assert "Docker Root Dir" in deploy_shell
    assert "/opt/dlami/nvme" in deploy_shell
    assert "DOCKER_DATA_ROOT=/opt/dlami/nvme/docker" in deploy_shell
    assert "sudo nvidia-ctk runtime configure" in deploy_shell
    assert "sudo tee /etc/docker/daemon.json" not in deploy_shell
    assert "chmod -R 777" not in deploy_shell
    assert "chmod -R u+rwX,g+rwX,o-rwx" in deploy_shell
    assert "docker login nvcr.io -u" in deploy_shell
    env_file_commands = [
        command.argv
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    ]
    assert len(env_file_commands) == 1
    env_file_shell = env_file_commands[0][2]
    credential_python = env_file_commands[0][6]
    remote_install_shell = env_file_commands[0][7]
    assert env_file_commands[0][5] == sys.executable
    assert env_file_commands[0][8] == str(REPO_ROOT)
    assert 'credential_env="$("$2" -c "$3" "$5")"' in env_file_shell
    assert (
        'printf "%s\\n" "$credential_env" | '
        "ssh -o BatchMode=yes -o ConnectTimeout=10 "
        "-o StrictHostKeyChecking=yes "
        '-o UserKnownHostsFile=~/.ssh/known_hosts "$1" "$4"' in env_file_shell
    )
    direct_ssh_commands = [
        command.argv
        for command in plan.deploy_commands
        if command.argv and command.argv[0] == "ssh"
    ]
    assert direct_ssh_commands
    for command in direct_ssh_commands:
        assert "StrictHostKeyChecking=yes" in command
        assert "UserKnownHostsFile=~/.ssh/known_hosts" in command
    assert "dotenv_values(dotenv_path=dotenv_path, interpolate=False)" in (
        credential_python
    )
    assert 'dotenv_path = Path(sys.argv[1]) / ".env"' in credential_python
    assert 'credential_names = ("NGC_API_KEY", "HF_TOKEN")' in credential_python
    assert 'if "\\n" in value or "\\r" in value' in credential_python
    assert 'tmp_env="$(mktemp "${destination}.XXXXXX")"' in remote_install_shell
    assert 'chmod 600 "$tmp_env"' in remote_install_shell
    assert 'mv -f -- "$tmp_env" "$destination"' in remote_install_shell
    assert "trap cleanup EXIT" in remote_install_shell
    assert 'destination="$HOME/.ngc-nim.env"' in remote_install_shell
    assert "set -e -o pipefail" in deploy_shell
    assert ". ./.env" not in deploy_shell
    assert "/tmp/wu-ngc-nim.env" not in deploy_shell
    assert '--env-file "$HOME/.ngc-nim.env"' in deploy_shell
    assert "brev copy" not in env_file_shell
    assert "/home/ubuntu/.env" not in deploy_shell
    login_command = next(
        command.argv[-1]
        for command in plan.deploy_commands
        if command.label == "Log Docker into nvcr.io without exposing the NGC token"
    )
    assert '. "$HOME/.ngc-nim.env"' not in login_command
    assert '"$HOME/.ngc-nim.env"' in login_command
    assert "sed -n 's/^NGC_API_KEY=//p'" in login_command
    assert "tr -d '\\r\\n'" in login_command
    assert 'test "$cleaned" = "$NGC_API_KEY"' in login_command
    assert "flux.2-klein-4b:1.0.1-variant" in deploy_shell
    image_readiness_commands = [
        command.argv
        for command in plan.deploy_commands
        if command.label
        == "Verify the image-generation endpoint is ready on the image-gen node"
    ]
    assert image_readiness_commands == [
        (
            "brev",
            "exec",
            "wu-test-texture-image-gen",
            "curl -fsS http://localhost:8000/v1/health/ready",
        )
    ]
    assert "Docker Root Dir" in deploy_shell
    assert "brev port-forward wu-test-texture-image-gen -p 8005:8000" in deploy_shell
    assert "brev port-forward wu-test-texture-llm" not in deploy_shell
    assert "http://localhost:8005/v1/health/ready" in deploy_shell
    assert (
        "apps/texture_agent_service/docker-compose.brev-host-llm.yml"
        not in deploy_shell
    )
    assert "TA_IMAGE_GEN_BACKEND=openai" in plan.environment
    assert "TA_IMAGE_GEN_BASE_URL=http://localhost:8005/v1" in plan.environment
    assert "TA_IMAGE_GEN_MODEL=black-forest-labs/flux.2-klein-4b" in plan.environment
    assert "TA_IMAGE_GEN_API_KEY=not-used" in plan.environment
    assert not any(entry.startswith("TA_LLM_") for entry in plan.environment)
    assert any("explicit prompts" in entry for entry in plan.environment)
    assert any("skipped by default" in note for note in plan.notes)
    assert any("Brev-hosted dependency endpoints" in note for note in plan.notes)


def test_texture_hybrid_can_target_an_exact_existing_image_gen_node() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        image_gen_node_name="content-image-gen",
    )
    (image_gen_node,) = plan.nodes

    assert image_gen_node.name == "content-image-gen"
    assert image_gen_node.dry_run.argv[2] == "content-image-gen"
    assert all(
        "wu-test-texture-image-gen" not in command.argv
        for command in plan.deploy_commands
    )
    assert any(
        command.argv[-2:] == ("content-image-gen", "true")
        for command in plan.deploy_commands
    )


@POSIX_SHELL_ONLY
def test_texture_hybrid_streams_credentials_without_a_local_secret_file(
    tmp_path: Path,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    captured_env = tmp_path / "captured.env"
    shell_script = (
        'ssh() { test "$9" = wu-test-texture-image-gen; '
        'cat > "$SSH_CAPTURE"; }; ' + env_file_command.argv[2]
    )
    environment = {
        **os.environ,
        "SSH_CAPTURE": str(captured_env),
        "HF_TOKEN": "test-hf-token",
        "NGC_API_KEY": "test-ngc-key",
        "TMPDIR": str(temp_dir),
    }
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert captured_env.read_text(encoding="utf-8") == (
        "NGC_API_KEY=test-ngc-key\nHF_TOKEN=test-hf-token\n"
    )
    assert list(temp_dir.iterdir()) == []

    failed = subprocess.run(
        (
            "bash",
            "-c",
            "ssh() { cat >/dev/null; return 9; }; " + env_file_command.argv[2],
            *command_args,
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    assert failed.returncode == 9


@pytest.mark.parametrize("missing_credential", ["NGC_API_KEY", "HF_TOKEN"])
@POSIX_SHELL_ONLY
def test_texture_hybrid_requires_each_credential_before_ssh(
    tmp_path: Path,
    missing_credential: str,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    ssh_marker = tmp_path / "ssh-called"
    shell_script = (
        'ssh() { : > "$SSH_MARKER"; cat >/dev/null; }; ' + env_file_command.argv[2]
    )
    environment = {
        **os.environ,
        "SSH_MARKER": str(ssh_marker),
        "HF_TOKEN": "test-hf-token",
        "NGC_API_KEY": "test-ngc-key",
    }
    environment.pop(missing_credential)
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    failed = subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    assert failed.returncode != 0
    assert not ssh_marker.exists()


@pytest.mark.parametrize("credential_name", ["NGC_API_KEY", "HF_TOKEN"])
@pytest.mark.parametrize("line_break", ["\n", "\r"], ids=["lf", "cr"])
@POSIX_SHELL_ONLY
def test_texture_hybrid_rejects_multiline_credential_before_ssh(
    tmp_path: Path,
    credential_name: str,
    line_break: str,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    ssh_marker = tmp_path / "ssh-called"
    shell_script = (
        'ssh() { : > "$SSH_MARKER"; cat >/dev/null; }; ' + env_file_command.argv[2]
    )
    environment = {
        **os.environ,
        "SSH_MARKER": str(ssh_marker),
        "HF_TOKEN": "test-hf_token",
        "NGC_API_KEY": "test-ngc_key+=:/@%",
    }
    environment[credential_name] = f"line-one{line_break}line-two"
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    failed = subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert failed.stderr.strip() == f"{credential_name} must be a single-line value"
    assert not ssh_marker.exists()


@POSIX_SHELL_ONLY
def test_texture_hybrid_prefers_exported_credentials_without_executing_dotenv(
    tmp_path: Path,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    captured_env = tmp_path / "captured.env"
    injection_marker = tmp_path / "dotenv-command-ran"
    (tmp_path / ".env").write_text(
        'NGC_API_KEY="ngc-!$\'(),;[]"\n'
        "HF_TOKEN='hf-!$\"(),;[]'\n"
        f"IGNORED=$(touch {injection_marker})\n",
        encoding="utf-8",
    )
    shell_script = 'ssh() { cat > "$SSH_CAPTURE"; }; ' + env_file_command.argv[2]
    environment = {
        **os.environ,
        "SSH_CAPTURE": str(captured_env),
        "HF_TOKEN": "environment-hf-token",
        "NGC_API_KEY": "environment-ngc-key",
    }
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert captured_env.read_text(encoding="utf-8") == (
        "NGC_API_KEY=environment-ngc-key\nHF_TOKEN=environment-hf-token\n"
    )
    assert not injection_marker.exists()


@POSIX_SHELL_ONLY
def test_texture_hybrid_uses_dotenv_when_credentials_are_not_exported(
    tmp_path: Path,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    captured_env = tmp_path / "captured.env"
    (tmp_path / ".env").write_text(
        "NGC_API_KEY=\"ngc-!$'(),;[]\"\nHF_TOKEN='hf-!$\"(),;[]'\n",
        encoding="utf-8",
    )
    shell_script = 'ssh() { cat > "$SSH_CAPTURE"; }; ' + env_file_command.argv[2]
    environment = {
        **os.environ,
        "SSH_CAPTURE": str(captured_env),
    }
    environment.pop("HF_TOKEN", None)
    environment.pop("NGC_API_KEY", None)
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=True,
    )

    assert captured_env.read_text(encoding="utf-8") == (
        "NGC_API_KEY=ngc-!$'(),;[]\nHF_TOKEN=hf-!$\"(),;[]\n"
    )


@pytest.mark.parametrize("credential_name", ["NGC_API_KEY", "HF_TOKEN"])
@POSIX_SHELL_ONLY
def test_texture_hybrid_rejects_an_explicit_empty_credential_without_dotenv_fallback(
    tmp_path: Path,
    credential_name: str,
) -> None:
    plan = _plan("texture", "hybrid")
    env_file_command = next(
        command
        for command in plan.deploy_commands
        if command.label == "Stream a minimal NGC/HF env file securely to FLUX NIM"
    )
    ssh_marker = tmp_path / "ssh-called"
    (tmp_path / ".env").write_text(
        "NGC_API_KEY=dotenv-ngc-key\nHF_TOKEN=dotenv-hf-token\n",
        encoding="utf-8",
    )
    shell_script = (
        'ssh() { : > "$SSH_MARKER"; cat >/dev/null; }; ' + env_file_command.argv[2]
    )
    environment = {
        **os.environ,
        "SSH_MARKER": str(ssh_marker),
        credential_name: "",
    }
    environment.pop(
        "HF_TOKEN" if credential_name == "NGC_API_KEY" else "NGC_API_KEY",
        None,
    )
    command_args = (*env_file_command.argv[3:-1], str(tmp_path))

    failed = subprocess.run(
        ("bash", "-c", shell_script, *command_args),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert failed.returncode != 0
    assert failed.stderr.strip() == f"{credential_name} is required"
    assert not ssh_marker.exists()


@POSIX_SHELL_ONLY
def test_remote_credential_install_is_mode_0600_and_symlink_safe(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "credentials.env"
    victim = tmp_path / "victim"
    injection_marker = tmp_path / "stdin-was-executed"
    victim.write_text("unchanged\n", encoding="utf-8")
    destination.symlink_to(victim)
    remote_shell = BREV.REMOTE_CREDENTIAL_INSTALL_SHELL.replace(
        'destination="$HOME/.ngc-nim.env"', f'destination="{destination}"'
    )
    payload = (
        f"NGC_API_KEY=ngc-!$;$(touch {injection_marker})[]\n"
        'HF_TOKEN=hf-`false`-$HOME-"quoted"\n'
    )

    subprocess.run(
        ("bash", "-c", remote_shell),
        input=payload,
        text=True,
        check=True,
    )

    assert not destination.is_symlink()
    assert destination.read_text(encoding="utf-8") == payload
    assert destination.stat().st_mode & 0o777 == 0o600
    assert victim.read_text(encoding="utf-8") == "unchanged\n"
    assert not injection_marker.exists()
    assert list(tmp_path.glob("credentials.env.*")) == []


def test_texture_hybrid_requires_python_dotenv_before_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find_spec = BREV.importlib.util.find_spec
    monkeypatch.setattr(
        BREV.importlib.util,
        "find_spec",
        lambda name: None if name == "dotenv" else original_find_spec(name),
    )

    with pytest.raises(RuntimeError, match="Activate the repository \\.venv"):
        _plan("texture", "hybrid")


def test_texture_hybrid_can_include_optional_small_qwen_llm() -> None:
    plan = _plan("texture", "hybrid", texture_include_llm=True)
    image_gen_node, llm_node = plan.nodes
    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert image_gen_node.name == "wu-test-texture-image-gen"
    assert "L4" in llm_node.dry_run.argv
    assert "24" in llm_node.dry_run.argv
    assert "df -h / /home /mnt/*" in deploy_shell
    assert "brev port-forward wu-test-texture-llm -p 8003:8000" in deploy_shell
    assert "TA_LLM_BASE_URL=http://localhost:8003/v1" in plan.environment
    assert "TA_LLM_MODEL=Qwen/Qwen3.5-4B" in plan.environment
    assert "TA_NIM_API_KEY=not-used" in plan.environment
    assert "TA_IMAGE_GEN_API_KEY=not-used" in plan.environment


def test_texture_hybrid_omits_placeholder_key_for_custom_image_gen_endpoint() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        image_gen_base_url="https://image-gen-tunnel.example.test/v1",
    )

    assert (
        "TA_IMAGE_GEN_BASE_URL=https://image-gen-tunnel.example.test/v1"
        in plan.environment
    )
    assert "TA_IMAGE_GEN_API_KEY=not-used" not in plan.environment
    assert any(
        "TA_IMAGE_GEN_API_KEY" in entry and "authenticated" in entry
        for entry in plan.environment
    )


def test_texture_single_host_sidecars_use_image_gen_api_key_placeholder() -> None:
    plan = _plan("texture", "single-host-local-sidecars")
    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert "apps/texture_agent_service/docker-compose.multi-gpu.yml" in deploy_shell
    assert "TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1" in plan.environment
    assert "TA_IMAGE_GEN_API_KEY=not-used" in plan.environment
    assert "TA_NIM_API_KEY=not-used" in plan.environment


def test_texture_single_host_sidecars_allow_non_rtx_model_gpus() -> None:
    plan = _plan(
        "texture",
        "single-host-local-sidecars",
        render_gpu_name="H100",
        render_min_vram_gb=80,
    )
    node = plan.nodes[0]

    assert "H100" in node.dry_run.argv
    assert "160" in node.dry_run.argv


def test_texture_hybrid_can_use_image_gen_gpu_search_filters() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        image_gen_type=None,
        image_gen_gpu_name="RTX 6000",
        image_gen_min_vram_gb=48,
    )
    (image_gen_node,) = plan.nodes

    assert "--gpu-name" in image_gen_node.dry_run.argv
    assert "RTX 6000" in image_gen_node.dry_run.argv
    assert "48" in image_gen_node.dry_run.argv


def test_texture_hybrid_can_override_llm_gpu_capacity() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        texture_include_llm=True,
        vlm_gpu_name="H100",
        vlm_min_vram_gb=80,
    )
    _, llm_node = plan.nodes

    assert "H100" in llm_node.dry_run.argv
    assert "80" in llm_node.dry_run.argv


def test_texture_hybrid_same_node_service_mode_uses_host_overlay() -> None:
    plan = _plan("texture", "hybrid", hybrid_connectivity="same-node")
    deploy_shell = "\n".join(command.shell() for command in plan.deploy_commands)

    assert "apps/texture_agent_service/docker-compose.brev-host-llm.yml" in deploy_shell
    assert "TA_LLM_BASE_URL=http://host.docker.internal:8000/v1" in plan.environment
    assert "TA_NIM_API_KEY=not-used" in plan.environment
    assert "TA_LLM_BASE_URL=http://host.docker.internal:8000/v1" in deploy_shell
    assert "TA_NIM_API_KEY=not-used" in deploy_shell
    assert "> .env" in deploy_shell
    assert (
        "docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml"
        in deploy_shell
    )


def test_texture_qwen_model_can_be_overridden() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        texture_include_llm=True,
        qwen_model="qwen/qwen3.5-72b-instruct",
    )

    assert "TA_LLM_MODEL=qwen/qwen3.5-72b-instruct" in plan.environment


def test_texture_hybrid_can_use_tunnel_endpoint_for_two_node_path() -> None:
    plan = _plan(
        "texture",
        "hybrid",
        hybrid_connectivity="tunnel-url",
        model_base_url="https://llm-tunnel.example.test/v1",
    )
    service_node, llm_node = plan.nodes

    assert service_node.role == "CPU texture service node"
    assert "L4" in llm_node.dry_run.argv
    assert "TA_LLM_BASE_URL=https://llm-tunnel.example.test/v1" in plan.environment
    assert "TA_NIM_API_KEY=not-used" not in plan.environment
    assert any("TA_NIM_API_KEY" in entry for entry in plan.environment)
    assert any("Plain brev port-forward" in note for note in plan.notes)


def test_invalid_preset_reports_supported_values() -> None:
    with pytest.raises(ValueError, match="Supported presets"):
        _plan("texture", "render-only")
