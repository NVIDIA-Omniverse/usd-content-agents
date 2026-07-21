# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest


def load_deploy_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "deploy" / "collection" / "deploy.py"
    spec = importlib.util.spec_from_file_location("collection_deploy", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collection_default_config_builds_required_env() -> None:
    deploy = load_deploy_module()
    config = deploy.load_config(
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "collection"
        / "collection.yaml"
    )

    env, errors = deploy.build_env(config)

    assert errors == []
    assert env["RENDER_ENDPOINT"] == "http://host.docker.internal:8001"
    assert env["COLLECTION_MATERIAL_PORT"] == "8100"
    assert env["COLLECTION_PHYSICS_PORT"] == "8200"
    assert env["COLLECTION_TEXTURE_PORT"] == "8300"
    assert env["MA_RENDERING_USE_DATA_URI"] == "true"
    assert env["PA_RENDER_BACKEND"] == "remote"


def test_collection_optional_model_endpoints_wire_agent_env() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True, "host_port": 8100},
            "physics": {"enabled": True, "host_port": 8200},
            "texture": {"enabled": True, "host_port": 8300},
        },
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://render.example:8001",
            },
            "vlm": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://vlm.example:8000/v1",
                "backend": "nim",
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "temperature": 0.2,
                "max_tokens": 512,
            },
            "llm": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://llm.example:8000/v1",
                "backend": "nim",
                "model": "nvidia/llama-3.1-nemotron-nano-8b-v1",
                "temperature": 0.1,
                "max_tokens": 256,
            },
            "image_gen": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://image.example:8000/v1",
                "backend": "openai",
                "model": "black-forest-labs/flux.2-klein-4b",
                "api_key_env": "IMAGE_GEN_API_KEY",
            },
            "embeddings": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://embed.example:8000/v1",
                "backend": "nim",
                "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                "api_key_env": "EMBEDDING_API_KEY",
            },
        },
    }

    env, errors = deploy.build_env(config)

    assert errors == []
    assert env["MA_VLM_NIM_BASE_URL"] == "http://vlm.example:8000/v1"
    assert env["PA_VLM_NIM_BASE_URL"] == "http://vlm.example:8000/v1"
    assert env["MA_VLM_TEMPERATURE"] == "0.2"
    assert env["PA_VLM_TEMPERATURE"] == "0.2"
    assert env["MA_VLM_MAX_TOKENS"] == "512"
    assert env["PA_VLM_MAX_TOKENS"] == "512"
    assert env["MA_LLM_NIM_BASE_URL"] == "http://llm.example:8000/v1"
    assert env["TA_LLM_BASE_URL"] == "http://llm.example:8000/v1"
    assert env["MA_LLM_TEMPERATURE"] == "0.1"
    assert env["MA_LLM_MAX_TOKENS"] == "256"
    assert env["TA_IMAGE_GEN_BACKEND"] == "openai"
    assert env["MA_IMAGE_GEN_BACKEND"] == "openai"
    assert env["TA_IMAGE_GEN_MODEL"] == "black-forest-labs/flux.2-klein-4b"
    assert env["MA_IMAGE_GEN_MODEL"] == "black-forest-labs/flux.2-klein-4b"
    assert env["TA_IMAGE_GEN_BASE_URL"] == "http://image.example:8000/v1"
    assert env["MA_IMAGE_GEN_BASE_URL"] == "http://image.example:8000/v1"
    assert env["TA_IMAGE_GEN_API_KEY_ENV"] == "IMAGE_GEN_API_KEY"
    assert env["MA_IMAGE_GEN_API_KEY_ENV"] == "IMAGE_GEN_API_KEY"
    assert "TA_IMAGE_GEN_API_KEY" not in env
    assert "MA_IMAGE_GEN_API_KEY" not in env
    assert env["MA_CLUSTER_EMBEDDING_BASE_URL"] == "http://embed.example:8000/v1"
    assert env["MA_CLUSTER_EMBEDDING_API_KEY_ENV"] == "EMBEDDING_API_KEY"
    assert "MA_CLUSTER_EMBEDDING_API_KEY" not in env


def test_collection_openai_vlm_base_url_does_not_emit_nim_routing() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True, "host_port": 8100},
            "physics": {"enabled": True, "host_port": 8200},
            "texture": {"enabled": False},
        },
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://render.example:8001",
            },
            "vlm": {
                "enabled": True,
                "provider": "external",
                "backend": "openai",
                "model": "my-custom-vlm",
                "base_url": "https://api.openai-compatible.example/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
    }

    env, errors = deploy.build_env(config)

    assert errors == []
    assert env["MA_VLM_BACKEND"] == "openai"
    assert env["PA_VLM_BACKEND"] == "openai"
    assert env["MA_VLM_BASE_URL"] == "https://api.openai-compatible.example/v1"
    assert env["PA_VLM_BASE_URL"] == "https://api.openai-compatible.example/v1"
    assert env["MA_VLM_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert env["PA_VLM_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert "MA_VLM_NIM_BASE_URL" not in env
    assert "PA_VLM_NIM_BASE_URL" not in env
    assert "MA_NIM_API_KEY" not in env
    assert "PA_NIM_API_KEY" not in env


def test_collection_openai_llm_base_url_uses_generic_endpoint_env() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True, "host_port": 8100},
            "physics": {"enabled": False},
            "texture": {"enabled": True, "host_port": 8300},
        },
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://render.example:8001",
            },
            "llm": {
                "enabled": True,
                "provider": "external",
                "backend": "openai",
                "model": "my-custom-llm",
                "endpoint": "https://api.openai-compatible.example/v1",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
    }

    env, errors = deploy.build_env(config)

    assert errors == []
    assert env["MA_LLM_BACKEND"] == "openai"
    assert env["TA_LLM_BACKEND"] == "openai"
    assert env["MA_LLM_BASE_URL"] == "https://api.openai-compatible.example/v1"
    assert env["TA_LLM_BASE_URL"] == "https://api.openai-compatible.example/v1"
    assert env["MA_LLM_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert env["TA_LLM_API_KEY_ENV"] == "OPENAI_API_KEY"
    assert "MA_LLM_NIM_BASE_URL" not in env
    assert "MA_NIM_API_KEY" not in env


def test_collection_openai_vlm_base_url_requires_paired_key() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True, "host_port": 8100},
            "physics": {"enabled": True, "host_port": 8200},
            "texture": {"enabled": False},
        },
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://render.example:8001",
            },
            "vlm": {
                "enabled": True,
                "provider": "external",
                "backend": "openai",
                "model": "my-custom-vlm",
                "base_url": "https://api.openai-compatible.example/v1",
            },
        },
    }

    _env, errors = deploy.build_env(config)

    assert any("dependencies.vlm.api_key or api_key_env" in error for error in errors)


def test_collection_openai_llm_base_url_requires_paired_key() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True, "host_port": 8100},
            "physics": {"enabled": False},
            "texture": {"enabled": True, "host_port": 8300},
        },
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://render.example:8001",
            },
            "llm": {
                "enabled": True,
                "provider": "external",
                "backend": "openai",
                "model": "my-custom-llm",
                "base_url": "https://api.openai-compatible.example/v1",
            },
        },
    }

    _env, errors = deploy.build_env(config)

    assert any("dependencies.llm.api_key or api_key_env" in error for error in errors)


def test_collection_docs_describe_shared_image_gen_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required_phrases = [
        "Texture Agent texture generation",
        "Material Agent-generated reference images",
        "Material Agent-generated material-library textures",
    ]
    docs = [
        repo_root / "deploy" / "collection" / "README.md",
        repo_root / "deploy" / "collection" / "collection.yaml",
        repo_root / ".agents" / "skills" / "deploy-collection" / "SKILL.md",
        repo_root
        / ".agents"
        / "skills"
        / "deploy-collection"
        / "references"
        / "env-contract.md",
        repo_root
        / ".agents"
        / "skills"
        / "deploy-collection"
        / "references"
        / "topologies.md",
    ]

    for path in docs:
        text = " ".join(path.read_text(encoding="utf-8").split())
        for phrase in required_phrases:
            assert phrase in text, path


def test_collection_env_preview_redacts_sensitive_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    image_api_key = "YOUR_IMAGE_API_KEY_HERE"
    embedding_api_key = "YOUR_EMBEDDING_API_KEY_HERE"
    config = {
        "agents": {
            "material": {"enabled": False},
            "physics": {"enabled": False},
            "texture": {"enabled": True},
        },
        "dependencies": {
            "render": {"enabled": False},
            "image_gen": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://image.example:8000/v1",
                "api_key": image_api_key,
            },
            "embeddings": {
                "enabled": True,
                "provider": "external",
                "endpoint": "http://embed.example:8000/v1",
                "api_key": embedding_api_key,
            },
        },
    }

    env, errors = deploy.build_env(config)

    assert errors == []
    redacted = deploy.env_text(env)
    unredacted = deploy.env_text(env, redact=False)
    assert image_api_key not in redacted
    assert embedding_api_key not in redacted
    assert "TA_IMAGE_GEN_API_KEY=REDACTED" in redacted
    assert "MA_CLUSTER_EMBEDDING_API_KEY=REDACTED" in redacted
    assert f"TA_IMAGE_GEN_API_KEY={image_api_key}" in unredacted

    assert deploy.print_plan(config) == 0
    output = capsys.readouterr().out
    assert image_api_key not in output
    assert embedding_api_key not in output
    assert "Secrets redacted" in output


def test_collection_write_env_allows_output_outside_repo(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    config = deploy.load_config(
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "collection"
        / "collection.yaml"
    )
    output = tmp_path / "collection.env"

    deploy.write_env(config, output)

    assert output.exists()
    assert str(output) in capsys.readouterr().out


def test_collection_missing_render_endpoint_is_error_for_rendering_agents() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": True},
            "physics": {"enabled": False},
            "texture": {"enabled": True},
        },
        "dependencies": {"render": {"enabled": True, "endpoint": ""}},
    }

    _, errors = deploy.build_env(config)

    assert errors == [
        "dependencies.render.endpoint is required when material or physics is enabled"
    ]


def test_collection_enabled_services_honors_agent_flags() -> None:
    deploy = load_deploy_module()
    config = {
        "agents": {
            "material": {"enabled": False},
            "physics": {"enabled": False},
            "texture": {"enabled": True},
        },
        "dependencies": {"render": {"enabled": False, "endpoint": ""}},
    }

    env, errors = deploy.build_env(config)

    assert errors == []
    assert "RENDER_ENDPOINT" not in env
    assert deploy.enabled_agent_services(config) == ["texture-agent-service"]


def test_collection_fetches_scene_optimizer_resources_for_material_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    package_dir = tmp_path / "scene_optimizer_core"
    fetch_script = tmp_path / "fetch_build_resources.sh"
    fetch_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(deploy, "SCENE_OPTIMIZER_BUILD_RESOURCES", package_dir)
    monkeypatch.setattr(deploy, "FETCH_BUILD_RESOURCES", fetch_script)
    calls: list[list[str]] = []

    rc = deploy.ensure_scene_optimizer_build_resources(
        ["material-agent-service"],
        runner=lambda command: calls.append(command) or 0,
    )

    assert rc == 0
    assert calls == [[str(fetch_script)]]
    assert "Fetching Scene Optimizer Core build resources" in capsys.readouterr().out


def test_collection_verifies_scene_optimizer_resources_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    package_dir = tmp_path / "scene_optimizer_core"
    for subdir in deploy.SCENE_OPTIMIZER_REQUIRED_SUBDIRS:
        (package_dir / subdir).mkdir(parents=True)
    fetch_script = tmp_path / "fetch_build_resources.sh"
    fetch_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    monkeypatch.setattr(deploy, "SCENE_OPTIMIZER_BUILD_RESOURCES", package_dir)
    monkeypatch.setattr(deploy, "FETCH_BUILD_RESOURCES", fetch_script)
    calls: list[list[str]] = []

    rc = deploy.ensure_scene_optimizer_build_resources(
        ["physics-agent-service"],
        runner=lambda command: calls.append(command) or 0,
    )

    assert rc == 0
    assert calls == [[str(fetch_script)]]
    assert "Validating Scene Optimizer Core build resources" in capsys.readouterr().out


def test_collection_missing_scene_optimizer_fetch_script_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()
    monkeypatch.setattr(deploy, "FETCH_BUILD_RESOURCES", tmp_path / "missing.sh")
    calls: list[list[str]] = []

    rc = deploy.ensure_scene_optimizer_build_resources(
        ["material-agent-service"],
        runner=lambda command: calls.append(command) or 0,
    )

    assert rc == 1
    assert calls == []


def test_collection_up_continues_when_scene_optimizer_fetch_fails_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        """
agents:
  material:
    enabled: true
    host_port: 8100
  physics:
    enabled: false
  texture:
    enabled: false
dependencies:
  render:
    enabled: true
    provider: external
    endpoint: http://render.example:8001
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(deploy, "GENERATED_ENV", tmp_path / ".collection.generated.env")
    monkeypatch.setattr(
        deploy,
        "ensure_scene_optimizer_build_resources",
        lambda services: 7,
    )
    compose_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        deploy,
        "run_compose",
        lambda config, *args: compose_calls.append((config, *args)) or 0,
    )

    rc = deploy.main(["-c", str(config_path), "up"])

    assert rc == 0
    assert len(compose_calls) == 1
    assert (
        "Scene Optimizer Core build-resource preflight failed"
        in capsys.readouterr().out
    )


def test_collection_up_stops_when_required_scene_optimizer_fetch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()
    config_path = tmp_path / "collection.yaml"
    config_path.write_text(
        """
agents:
  material:
    enabled: true
    host_port: 8100
  physics:
    enabled: false
  texture:
    enabled: false
dependencies:
  render:
    enabled: true
    provider: external
    endpoint: http://render.example:8001
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(deploy, "GENERATED_ENV", tmp_path / ".collection.generated.env")
    monkeypatch.setattr(
        deploy,
        "ensure_scene_optimizer_build_resources",
        lambda services: 7,
    )
    compose_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        deploy,
        "run_compose",
        lambda config, *args: compose_calls.append((config, *args)) or 0,
    )

    rc = deploy.main(["-c", str(config_path), "up", "--require-local-scene-optimizer"])

    assert rc == 7
    assert compose_calls == []


def test_collection_texture_only_build_does_not_fetch_scene_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()
    monkeypatch.setattr(deploy, "SCENE_OPTIMIZER_BUILD_RESOURCES", tmp_path / "missing")
    calls: list[list[str]] = []

    rc = deploy.ensure_scene_optimizer_build_resources(
        ["texture-agent-service"],
        runner=lambda command: calls.append(command) or 0,
    )

    assert rc == 0
    assert calls == []


def test_collection_texture_only_compose_config_does_not_require_render_endpoint(
    tmp_path: Path,
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not available")

    repo_root = Path(__file__).resolve().parents[1]
    compose_file = repo_root / "deploy" / "collection" / "docker-compose.agents.yml"
    env = os.environ.copy()
    env.pop("RENDER_ENDPOINT", None)
    env["COMPOSE_DISABLE_ENV_FILE"] = "1"

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(tmp_path),
            "-f",
            str(compose_file),
            "config",
            "--quiet",
            "texture-agent-service",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr


def test_collection_host_check_url_maps_container_host_gateway() -> None:
    deploy = load_deploy_module()

    assert (
        deploy.host_check_url("http://host.docker.internal:8013/health")
        == "http://localhost:8013/health"
    )
    assert (
        deploy.host_check_url("http://host.docker.internal:8014/v1/health/ready")
        == "http://localhost:8014/v1/health/ready"
    )
    assert (
        deploy.host_check_url("http://render.example:8001/health")
        == "http://render.example:8001/health"
    )


def test_collection_render_smoke_requires_initialized_ovrtx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()

    monkeypatch.setattr(
        deploy,
        "try_json_url",
        lambda url, timeout=5.0: (
            True,
            "HTTP 200",
            {"status": "initializing", "gpu_initialized": False},
        ),
    )

    assert (
        deploy.check_render_url("render", "http://localhost:8013/health", strict=False)
        is True
    )
    assert (
        deploy.check_render_url("render", "http://localhost:8013/health", strict=True)
        is False
    )


def test_collection_render_smoke_rejects_missing_gpu_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()

    for payload in (None, [], {"status": "healthy"}, {"gpu_initialized": 0}):
        monkeypatch.setattr(
            deploy,
            "try_json_url",
            lambda url, timeout=5.0, payload=payload: (
                True,
                "HTTP 200",
                payload,
            ),
        )

        assert (
            deploy.check_render_url(
                "render", "http://localhost:8013/health", strict=True
            )
            is False
        )

    monkeypatch.setattr(
        deploy,
        "try_json_url",
        lambda url, timeout=5.0: (
            True,
            "HTTP 200",
            {"status": "healthy", "gpu_initialized": True},
        ),
    )

    assert (
        deploy.check_render_url("render", "http://localhost:8013/health", strict=True)
        is True
    )


def test_collection_examples_build_env_without_errors() -> None:
    deploy = load_deploy_module()
    examples_dir = (
        Path(__file__).resolve().parents[1] / "deploy" / "collection" / "examples"
    )

    for path in sorted(examples_dir.glob("*.yaml")):
        config = deploy.load_config(path)
        env, errors = deploy.build_env(config)

        assert errors == [], path
        assert "RENDER_ENDPOINT" in env, path


def test_collection_brev_plan_uses_expected_roles_and_ports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    config = deploy.load_config(
        Path(__file__).resolve().parents[1]
        / "deploy"
        / "collection"
        / "examples"
        / "full-brev.yaml"
    )

    assert deploy.enabled_brev_roles(config) == [
        "render",
        "vlm",
        "image_gen",
        "embeddings",
    ]
    assert deploy.print_brev_plan(config) == 0

    output = capsys.readouterr().out
    assert "brev create content-render --type g6e.xlarge" in output
    assert "brev port-forward content-render -p 8013:8001" in output
    assert "brev port-forward content-vlm -p 8015:8000" in output
    assert "dependencies.embeddings.endpoint: http://host.docker.internal:8014/v1" in (
        output
    )
    assert "LLM note" in output


def test_collection_brev_provision_reuses_existing_and_creates_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    deploy = load_deploy_module()
    config = {
        "dependencies": {
            "render": {
                "enabled": True,
                "provider": "brev",
                "endpoint": "http://host.docker.internal:8013",
            },
            "embeddings": {
                "enabled": True,
                "provider": "brev",
                "endpoint": "http://host.docker.internal:8014/v1",
            },
        }
    }
    commands: list[list[str]] = []

    monkeypatch.setattr(
        deploy,
        "brev_list_instances",
        lambda: {"content-render": {"name": "content-render"}},
    )

    assert (
        deploy.provision_brev(
            config,
            execute=True,
            runner=lambda command: commands.append(command) or 0,
        )
        == 0
    )

    assert commands == [
        ["brev", "create", "content-embeddings", "--type", "g6e.xlarge"]
    ]
    output = capsys.readouterr().out
    assert "[reuse] render: content-render" in output
    assert "brev port-forward content-embeddings -p 8014:8004" in output


def test_collection_brev_list_instances_accepts_workspaces_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deploy = load_deploy_module()

    def fake_run(
        command: Sequence[str],
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["brev", "ls", "--json"]
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"workspaces":[{"name":"content-render"},{"id":"abc123"}]}',
            stderr="",
        )

    monkeypatch.setattr(deploy.subprocess, "run", fake_run)

    assert deploy.brev_list_instances() == {
        "content-render": {"name": "content-render"},
        "abc123": {"id": "abc123"},
    }


def test_collection_optional_sidecar_compose_files_parse() -> None:
    deploy = load_deploy_module()
    collection_dir = Path(__file__).resolve().parents[1] / "deploy" / "collection"

    for filename in (
        "docker-compose.vlm.yml",
        "docker-compose.llm.yml",
        "docker-compose.embeddings.yml",
        "docker-compose.image-gen.yml",
    ):
        data = deploy.load_config(collection_dir / filename)

        assert "services" in data, filename


def test_collection_vllm_healthchecks_use_python3() -> None:
    deploy = load_deploy_module()
    collection_dir = Path(__file__).resolve().parents[1] / "deploy" / "collection"
    expected_command = (
        "python3 - <<'PY'\n"
        "import urllib.request\n"
        "urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)\n"
        "PY"
    )

    for filename, service_name in (
        ("docker-compose.vlm.yml", "vlm-vllm"),
        ("docker-compose.llm.yml", "llm-vllm"),
    ):
        data = deploy.load_config(collection_dir / filename)
        healthcheck = data["services"][service_name]["healthcheck"]["test"]

        assert healthcheck == ["CMD-SHELL", expected_command], filename


def test_collection_vlm_compose_matches_brev_runtime_path() -> None:
    deploy = load_deploy_module()
    collection_dir = Path(__file__).resolve().parents[1] / "deploy" / "collection"
    data = deploy.load_config(collection_dir / "docker-compose.vlm.yml")

    service = data["services"]["vlm-vllm"]
    command = service["command"]

    assert service["runtime"] == "${COLLECTION_VLM_DOCKER_RUNTIME:-nvidia}"
    assert "deploy" not in service
    assert "--served-model-name" in command
    assert "--dtype" in command
    assert "--trust-remote-code" in command
    assert (
        "${COLLECTION_VLM_CACHE_VOLUME:-vlm-vllm-cache}:/root/.cache/huggingface"
        in service["volumes"]
    )


def test_collection_nim_compose_allows_writable_cache_mounts() -> None:
    deploy = load_deploy_module()
    collection_dir = Path(__file__).resolve().parents[1] / "deploy" / "collection"

    embeddings = deploy.load_config(collection_dir / "docker-compose.embeddings.yml")
    image_gen = deploy.load_config(collection_dir / "docker-compose.image-gen.yml")

    assert (
        "${COLLECTION_EMBEDDINGS_CACHE_VOLUME:-embeddings-nim-cache}:/opt/nim/.cache"
        in embeddings["services"]["embeddings-nim"]["volumes"]
    )
    assert (
        "${COLLECTION_IMAGE_GEN_CACHE_VOLUME:-image-gen-nim-cache}:/opt/nim/.cache"
        in image_gen["services"]["image-gen-nim"]["volumes"]
    )


def test_collection_nim_compose_limits_runtime_secrets() -> None:
    deploy = load_deploy_module()
    collection_dir = Path(__file__).resolve().parents[1] / "deploy" / "collection"

    sidecars = {
        "docker-compose.embeddings.yml": "embeddings-nim",
        "docker-compose.image-gen.yml": "image-gen-nim",
    }
    for filename, service_name in sidecars.items():
        data = deploy.load_config(collection_dir / filename)
        service = data["services"][service_name]

        assert "env_file" not in service, filename
        assert "NGC_API_KEY=${NGC_API_KEY:-}" in service["environment"], filename
        assert "HF_TOKEN=${HF_TOKEN:-}" in service["environment"], filename
