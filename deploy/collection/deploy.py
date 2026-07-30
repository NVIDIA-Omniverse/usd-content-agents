#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint-driven deployment helper for the Content Agents collection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "collection.yaml"
GENERATED_ENV = HERE / ".collection.generated.env"
COMPOSE_FILE = HERE / "docker-compose.agents.yml"
FETCH_BUILD_RESOURCES = REPO_ROOT / "scripts" / "fetch_build_resources.sh"
SCENE_OPTIMIZER_BUILD_RESOURCES = (
    REPO_ROOT / ".build-resources" / "scene_optimizer_core"
)
SCENE_OPTIMIZER_REQUIRED_SUBDIRS = ("python", "lib", "extraLibs", "usdpy")
AGENT_SERVICES = {
    "material": "material-agent-service",
    "physics": "physics-agent-service",
    "texture": "texture-agent-service",
}
SCENE_OPTIMIZER_AGENT_SERVICES = {
    "material-agent-service",
    "physics-agent-service",
}
BREV_ROLE_DEFAULTS = {
    "render": {
        "name": "content-render",
        "type": "g6e.xlarge",
        "local_port": 8013,
        "remote_port": 8001,
        "endpoint_suffix": "",
    },
    "vlm": {
        "name": "content-vlm",
        "type": "denvr_A100_sxm4_80G",
        "local_port": 8015,
        "remote_port": 8000,
        "endpoint_suffix": "/v1",
    },
    "llm": {
        "name": "content-llm",
        "type": "g6e.xlarge",
        "local_port": 8017,
        "remote_port": 8000,
        "endpoint_suffix": "/v1",
    },
    "image_gen": {
        "name": "content-image-gen",
        "type": "g6e.xlarge",
        "local_port": 8016,
        "remote_port": 8000,
        "endpoint_suffix": "/v1",
    },
    "embeddings": {
        "name": "content-embeddings",
        "type": "g6e.xlarge",
        "local_port": 8014,
        "remote_port": 8004,
        "endpoint_suffix": "/v1",
    },
}
SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def is_enabled(data: dict[str, Any], default: bool = False) -> bool:
    return bool(data.get("enabled", default))


def dependency(config: dict[str, Any], name: str) -> dict[str, Any]:
    deps = config.get("dependencies", {})
    if not isinstance(deps, dict):
        return {}
    value = deps.get(name, {})
    return value if isinstance(value, dict) else {}


def agent(config: dict[str, Any], name: str) -> dict[str, Any]:
    agents = config.get("agents", {})
    if not isinstance(agents, dict):
        return {}
    value = agents.get(name, {})
    return value if isinstance(value, dict) else {}


def endpoint(dep: dict[str, Any]) -> str:
    value = str(dep.get("endpoint") or "").strip()
    return value.rstrip("/")


def model_base_url(dep: dict[str, Any]) -> str:
    value = setting(dep, "base_url") or endpoint(dep)
    return value.rstrip("/")


def provider(dep: dict[str, Any]) -> str:
    return str(dep.get("provider") or "external").strip().lower()


def model(dep: dict[str, Any], default: str) -> str:
    return str(dep.get("model") or default).strip()


def backend(dep: dict[str, Any], default: str) -> str:
    return str(dep.get("backend") or default).strip()


def api_key(dep: dict[str, Any]) -> str:
    return str(dep.get("api_key") or "").strip()


def api_key_env(dep: dict[str, Any]) -> str:
    return setting(dep, "api_key_env")


def setting(dep: dict[str, Any], name: str) -> str:
    value = dep.get(name)
    if value is None:
        return ""
    return str(value).strip()


def set_if_value(env: dict[str, str], key: str, value: str | None) -> None:
    if value:
        env[key] = value


def set_model_endpoint_env(
    env: dict[str, str],
    prefixes: tuple[str, ...],
    role: str,
    dep: dict[str, Any],
    base_url: str,
) -> None:
    for prefix in prefixes:
        env[f"{prefix}_{role}_BASE_URL"] = base_url
        set_if_value(env, f"{prefix}_{role}_API_KEY", api_key(dep))
        set_if_value(env, f"{prefix}_{role}_API_KEY_ENV", api_key_env(dep))


def brev_settings(config: dict[str, Any]) -> dict[str, Any]:
    data = config.get("brev", {})
    return data if isinstance(data, dict) else {}


def endpoint_port(dep: dict[str, Any]) -> int | None:
    dep_endpoint = endpoint(dep)
    if not dep_endpoint:
        return None
    parsed = urlparse(dep_endpoint)
    return parsed.port


def enabled_brev_roles(config: dict[str, Any]) -> list[str]:
    roles: list[str] = []
    instances = brev_settings(config).get("instances", {})
    instances = instances if isinstance(instances, dict) else {}
    for role in BREV_ROLE_DEFAULTS:
        data = dependency(config, role)
        if is_enabled(data, role == "render") and provider(data) == "brev":
            if role == "llm" and role not in instances:
                vlm = dependency(config, "vlm")
                if provider(vlm) == "brev" and (
                    not endpoint(data) or endpoint(data) == endpoint(vlm)
                ):
                    continue
            roles.append(role)
    return roles


def brev_role_spec(config: dict[str, Any], role: str) -> dict[str, Any]:
    defaults = BREV_ROLE_DEFAULTS[role]
    instances = brev_settings(config).get("instances", {})
    instances = instances if isinstance(instances, dict) else {}
    override = instances.get(role, {})
    override = override if isinstance(override, dict) else {}
    dep = dependency(config, role)

    spec = {**defaults, **override}
    inferred_local_port = endpoint_port(dep)
    if inferred_local_port is not None and "local_port" not in override:
        spec["local_port"] = inferred_local_port
    return spec


def expected_brev_endpoint(spec: dict[str, Any]) -> str:
    return (
        f"http://host.docker.internal:{spec['local_port']}"
        f"{spec.get('endpoint_suffix', '')}"
    )


def brev_list_instances() -> dict[str, dict[str, Any]]:
    result = subprocess.run(
        ["brev", "ls", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "brev ls --json failed")

    payload = json.loads(result.stdout or "[]")
    if isinstance(payload, dict):
        payload = payload.get("instances") or payload.get("workspaces") or []
    if not isinstance(payload, list):
        raise RuntimeError("brev ls --json returned an unexpected shape")

    instances: dict[str, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("Name") or item.get("id")
        if name:
            instances[str(name)] = item
    return instances


def build_env(config: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    env: dict[str, str] = {}
    errors: list[str] = []

    material = agent(config, "material")
    physics = agent(config, "physics")
    texture = agent(config, "texture")
    runtime = config.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    max_sessions = runtime.get("max_active_sessions", {})
    max_sessions = max_sessions if isinstance(max_sessions, dict) else {}

    if is_enabled(material, True):
        env["COLLECTION_MATERIAL_PORT"] = str(material.get("host_port", 8100))
        env["MA_MAX_ACTIVE_SESSIONS"] = str(max_sessions.get("material", 1))
    if is_enabled(physics, True):
        env["COLLECTION_PHYSICS_PORT"] = str(physics.get("host_port", 8200))
        env["PA_MAX_ACTIVE_SESSIONS"] = str(max_sessions.get("physics", 1))
    if is_enabled(texture, True):
        env["COLLECTION_TEXTURE_PORT"] = str(texture.get("host_port", 8300))
        env["TA_MAX_ACTIVE_SESSIONS"] = str(max_sessions.get("texture", 4))

    render = dependency(config, "render")
    render_endpoint = endpoint(render)
    if is_enabled(material, True) or is_enabled(physics, True):
        if render_endpoint:
            env["RENDER_ENDPOINT"] = render_endpoint
            env["MA_RENDERING_USE_DATA_URI"] = "true"
            env["PA_RENDER_BACKEND"] = "remote"
            env["WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS"] = str(
                runtime.get("render_concurrency", 1)
            )
        else:
            errors.append(
                "dependencies.render.endpoint is required when material or "
                "physics is enabled"
            )

    vlm = dependency(config, "vlm")
    if is_enabled(vlm):
        vlm_base_url = model_base_url(vlm)
        vlm_backend = backend(vlm, "nim")
        vlm_model = model(vlm, "google/gemma-4-31b-it")
        env["MA_VLM_BACKEND"] = vlm_backend
        env["PA_VLM_BACKEND"] = vlm_backend
        env["MA_VLM_MODEL"] = vlm_model
        env["PA_VLM_MODEL"] = vlm_model
        set_if_value(env, "MA_VLM_MAX_TOKENS", setting(vlm, "max_tokens"))
        set_if_value(env, "PA_VLM_MAX_TOKENS", setting(vlm, "max_tokens"))
        set_if_value(env, "MA_VLM_TEMPERATURE", setting(vlm, "temperature"))
        set_if_value(env, "PA_VLM_TEMPERATURE", setting(vlm, "temperature"))
        if vlm_base_url and vlm_backend == "nim":
            env["MA_VLM_NIM_BASE_URL"] = vlm_base_url
            env["PA_VLM_NIM_BASE_URL"] = vlm_base_url
            env["MA_NIM_API_KEY"] = api_key(vlm) or "not-used"
            env["PA_NIM_API_KEY"] = api_key(vlm) or "not-used"
        elif vlm_base_url and vlm_backend == "openai":
            set_model_endpoint_env(env, ("MA", "PA"), "VLM", vlm, vlm_base_url)
            if not api_key(vlm) and not api_key_env(vlm):
                errors.append(
                    "dependencies.vlm.api_key or api_key_env is required when "
                    "backend=openai uses base_url"
                )
        elif provider(vlm) in {"brev", "local"}:
            errors.append("dependencies.vlm.endpoint is required after provisioning")

    llm = dependency(config, "llm")
    if is_enabled(llm):
        llm_base_url = model_base_url(llm)
        llm_backend = backend(llm, "nim")
        llm_model = model(llm, "google/gemma-4-31b-it")
        env["MA_LLM_BACKEND"] = llm_backend
        env["TA_LLM_BACKEND"] = llm_backend
        env["MA_LLM_MODEL"] = llm_model
        env["TA_LLM_MODEL"] = llm_model
        set_if_value(env, "MA_LLM_MAX_TOKENS", setting(llm, "max_tokens"))
        set_if_value(env, "MA_LLM_TEMPERATURE", setting(llm, "temperature"))
        if llm_base_url and llm_backend == "nim":
            env["MA_LLM_NIM_BASE_URL"] = llm_base_url
            env["TA_LLM_BASE_URL"] = llm_base_url
            env["MA_NIM_API_KEY"] = api_key(llm) or env.get(
                "MA_NIM_API_KEY", "not-used"
            )
        elif llm_base_url and llm_backend == "openai":
            set_model_endpoint_env(env, ("MA",), "LLM", llm, llm_base_url)
            env["TA_LLM_BASE_URL"] = llm_base_url
            set_if_value(env, "TA_LLM_API_KEY", api_key(llm))
            set_if_value(env, "TA_LLM_API_KEY_ENV", api_key_env(llm))
            if not api_key(llm) and not api_key_env(llm):
                errors.append(
                    "dependencies.llm.api_key or api_key_env is required when "
                    "backend=openai uses base_url"
                )
        elif provider(llm) in {"brev", "local"}:
            errors.append("dependencies.llm.endpoint is required after provisioning")

    image_gen = dependency(config, "image_gen")
    if is_enabled(image_gen):
        image_endpoint = endpoint(image_gen)
        image_backend = backend(image_gen, "openai")
        image_model = model(image_gen, "black-forest-labs/flux.2-klein-4b")
        env["TA_IMAGE_GEN_BACKEND"] = image_backend
        env["TA_IMAGE_GEN_MODEL"] = image_model
        env["MA_IMAGE_GEN_BACKEND"] = image_backend
        env["MA_IMAGE_GEN_MODEL"] = image_model
        set_if_value(env, "TA_IMAGE_GEN_API_KEY", api_key(image_gen))
        set_if_value(env, "TA_IMAGE_GEN_API_KEY_ENV", api_key_env(image_gen))
        set_if_value(env, "MA_IMAGE_GEN_API_KEY", api_key(image_gen))
        set_if_value(env, "MA_IMAGE_GEN_API_KEY_ENV", api_key_env(image_gen))
        if image_endpoint:
            env["TA_IMAGE_GEN_BASE_URL"] = image_endpoint
            env["MA_IMAGE_GEN_BASE_URL"] = image_endpoint
        elif provider(image_gen) in {"brev", "local"}:
            errors.append(
                "dependencies.image_gen.endpoint is required after provisioning"
            )

    embeddings = dependency(config, "embeddings")
    if is_enabled(embeddings):
        embedding_endpoint = endpoint(embeddings)
        env["MA_CLUSTER_EMBEDDING_BACKEND"] = backend(embeddings, "nim")
        env["MA_CLUSTER_EMBEDDING_MODEL"] = model(
            embeddings, "nvidia/llama-nemotron-embed-vl-1b-v2"
        )
        set_if_value(env, "MA_CLUSTER_EMBEDDING_API_KEY", api_key(embeddings))
        set_if_value(env, "MA_CLUSTER_EMBEDDING_API_KEY_ENV", api_key_env(embeddings))
        if embedding_endpoint:
            env["MA_CLUSTER_EMBEDDING_BASE_URL"] = embedding_endpoint
        elif provider(embeddings) in {"brev", "local"}:
            errors.append(
                "dependencies.embeddings.endpoint is required after provisioning"
            )

    return env, errors


def is_sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(marker in normalized for marker in SENSITIVE_ENV_MARKERS)


def env_value_text(key: str, value: str, *, redact: bool) -> str:
    if redact and is_sensitive_env_key(key):
        return "REDACTED"
    return value


def env_text(env: dict[str, str], *, redact: bool = True) -> str:
    banner = "# Generated by deploy/collection/deploy.py."
    if redact:
        banner = f"{banner} Secrets redacted."
    lines = [
        banner,
    ]
    lines.extend(
        f"{key}={env_value_text(key, value, redact=redact)}"
        for key, value in sorted(env.items())
    )
    return "\n".join(lines) + "\n"


def write_env(config: dict[str, Any], output: Path = GENERATED_ENV) -> None:
    env, errors = build_env(config)
    if errors:
        raise ValueError("\n".join(errors))
    output.write_text(env_text(env, redact=False), encoding="utf-8")
    try:
        display_path = output.relative_to(REPO_ROOT)
    except ValueError:
        display_path = output
    print(f"Wrote {display_path}")


def compose_command(config: dict[str, Any], *args: str) -> list[str]:
    project = str(config.get("name") or "content-agents")
    return [
        "docker",
        "compose",
        "--env-file",
        str(GENERATED_ENV),
        "-p",
        project,
        "-f",
        str(COMPOSE_FILE),
        *args,
    ]


def enabled_agent_services(config: dict[str, Any]) -> list[str]:
    services: list[str] = []
    for name, service in AGENT_SERVICES.items():
        if is_enabled(agent(config, name), True):
            services.append(service)
    return services


def has_scene_optimizer_build_resources() -> bool:
    return all(
        (SCENE_OPTIMIZER_BUILD_RESOURCES / subdir).is_dir()
        for subdir in SCENE_OPTIMIZER_REQUIRED_SUBDIRS
    )


def services_need_scene_optimizer_build_resources(services: list[str]) -> bool:
    return any(service in SCENE_OPTIMIZER_AGENT_SERVICES for service in services)


def ensure_scene_optimizer_build_resources(
    services: list[str],
    runner: Callable[[list[str]], int] | None = None,
) -> int:
    if not services_need_scene_optimizer_build_resources(services):
        return 0
    if not FETCH_BUILD_RESOURCES.is_file():
        print(f"Missing build resource fetch script: {FETCH_BUILD_RESOURCES}")
        return 1

    if has_scene_optimizer_build_resources():
        print(
            "Validating Scene Optimizer Core build resources at "
            f"{SCENE_OPTIMIZER_BUILD_RESOURCES}"
        )
    else:
        print("Fetching Scene Optimizer Core build resources before Docker build...")

    runner = runner or (
        lambda command: subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    )
    return runner([str(FETCH_BUILD_RESOURCES)])


def run_compose(config: dict[str, Any], *args: str) -> int:
    command = compose_command(config, *args)
    print(" ".join(command))
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


def url_join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def host_check_url(url: str) -> str:
    """Map container-only hostnames to a host-reachable URL for status checks."""
    parsed = urlparse(url)
    if parsed.hostname != "host.docker.internal":
        return url

    netloc = "localhost"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc))


def try_url(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300, f"HTTP {response.status}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)


def try_json_url(url: str, timeout: float = 5.0) -> tuple[bool, str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
            detail = f"HTTP {response.status}"
            if not (200 <= response.status < 300):
                return False, detail, None
            if not body.strip():
                return True, detail, None
            try:
                return True, detail, json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return True, f"{detail}, non-JSON body", None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc), None


def check_url(label: str, url: str, timeout: float = 5.0) -> bool:
    ok, detail = try_url(url, timeout)
    if not ok:
        print(f"[FAIL] {label}: {url} ({detail})")
        return False
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}: {url}")
    return ok


def check_render_url(
    label: str, url: str, *, strict: bool = False, timeout: float = 5.0
) -> bool:
    ok, detail, payload = try_json_url(url, timeout)
    if not ok:
        print(f"[FAIL] {label}: {url} ({detail})")
        return False
    if strict and (not isinstance(payload, dict) or not payload.get("gpu_initialized")):
        print(f"[FAIL] {label}: {url} (gpu_initialized=false)")
        return False
    print(f"[OK] {label}: {url}")
    return True


def check_any_url(label: str, urls: list[str], timeout: float = 5.0) -> bool:
    failures: list[str] = []
    for url in urls:
        ok, detail = try_url(url, timeout)
        if ok:
            print(f"[OK] {label}: {url}")
            return True
        failures.append(f"{url} ({detail})")

    print(f"[FAIL] {label}: none of {len(urls)} endpoint checks passed")
    for failure in failures:
        print(f"  - {failure}")
    return False


def print_plan(config: dict[str, Any]) -> int:
    print(f"Deployment: {config.get('name', 'content-agents')}")
    print()
    print("Agents")
    for name in ("material", "physics", "texture"):
        data = agent(config, name)
        enabled = is_enabled(data, True)
        port = data.get("host_port", "disabled")
        print(f"- {name}: {'enabled' if enabled else 'disabled'} port={port}")

    print()
    print("Dependencies")
    for name in ("render", "vlm", "llm", "image_gen", "embeddings"):
        data = dependency(config, name)
        enabled = is_enabled(data, name == "render")
        dep_provider = provider(data)
        dep_endpoint = model_base_url(data) or endpoint(data) or "<not set>"
        print(
            f"- {name}: {'enabled' if enabled else 'disabled'} "
            f"provider={dep_provider} endpoint={dep_endpoint}"
        )
        if enabled and dep_provider == "brev":
            if name == "llm" and dep_endpoint == endpoint(dependency(config, "vlm")):
                print("  Brev option selected: sharing the VLM endpoint.")
            else:
                print(
                    "  Brev option selected: provision instance, port-forward, "
                    "then set endpoint."
                )

    env, errors = build_env(config)
    print()
    print("Generated env preview")
    print(env_text(env), end="")
    if errors:
        print()
        print("Config errors")
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


def print_brev_plan(config: dict[str, Any]) -> int:
    roles = enabled_brev_roles(config)
    if not roles:
        print("No enabled dependencies use provider=brev.")
        return 0

    print("Brev roles")
    for role in roles:
        spec = brev_role_spec(config, role)
        name = spec["name"]
        instance_type = spec["type"]
        local_port = spec["local_port"]
        remote_port = spec["remote_port"]
        configured_endpoint = endpoint(dependency(config, role)) or "<not set>"
        expected_endpoint = expected_brev_endpoint(spec)

        print(f"- {role}: instance={name} type={instance_type}")
        print(f"  create/reuse: brev create {name} --type {instance_type}")
        print(f"  forward: brev port-forward {name} -p {local_port}:{remote_port}")
        print(f"  expected endpoint: {expected_endpoint}")
        print(f"  configured endpoint: {configured_endpoint}")

    llm = dependency(config, "llm")
    if is_enabled(llm) and provider(llm) == "brev" and "llm" not in roles:
        print()
        print("LLM note")
        print("- No separate Brev LLM role is configured by default.")
        print("- Use the VLM endpoint for LLM, or add brev.instances.llm explicitly.")

    print()
    print("After forwards are running, use Docker-reachable endpoints:")
    for role in roles:
        spec = brev_role_spec(config, role)
        print(f"- dependencies.{role}.endpoint: {expected_brev_endpoint(spec)}")
    return 0


def provision_brev(
    config: dict[str, Any],
    *,
    execute: bool = False,
    runner: Callable[[list[str]], int] | None = None,
) -> int:
    roles = enabled_brev_roles(config)
    if not roles:
        print("No enabled dependencies use provider=brev.")
        return 0

    runner = runner or (lambda command: subprocess.run(command, check=False).returncode)

    if not execute:
        print("Dry run. Re-run with --execute to create missing Brev instances.")
        return print_brev_plan(config)

    try:
        instances = brev_list_instances()
    except Exception as exc:
        print(f"Failed to inspect Brev instances: {exc}")
        return 1

    ok = True
    for role in roles:
        spec = brev_role_spec(config, role)
        name = str(spec["name"])
        instance_type = str(spec["type"])
        if name in instances:
            print(f"[reuse] {role}: {name}")
            continue

        command = ["brev", "create", name, "--type", instance_type]
        print(f"[create] {' '.join(command)}")
        ok = runner(command) == 0 and ok

    print()
    print("Start forwards in separate terminals when instances are ready:")
    for role in roles:
        spec = brev_role_spec(config, role)
        print(
            f"brev port-forward {spec['name']} "
            f"-p {spec['local_port']}:{spec['remote_port']}"
        )

    return 0 if ok else 1


def status(config: dict[str, Any], *, strict: bool = False) -> int:
    ok = True
    for name, port in (
        ("material", agent(config, "material").get("host_port", 8100)),
        ("physics", agent(config, "physics").get("host_port", 8200)),
        ("texture", agent(config, "texture").get("host_port", 8300)),
    ):
        if is_enabled(agent(config, name), True):
            ok = check_url(name, f"http://localhost:{port}/health") and ok

    render = dependency(config, "render")
    if is_enabled(render, True) and endpoint(render):
        ok = (
            check_render_url(
                "render",
                host_check_url(url_join(endpoint(render), "/health")),
                strict=strict,
            )
            and ok
        )

    for name in ("vlm", "llm", "image_gen", "embeddings"):
        data = dependency(config, name)
        if is_enabled(data) and model_base_url(data):
            base = model_base_url(data)
            if base.endswith("/v1"):
                urls = [
                    url_join(base, "/health/ready"),
                    url_join(base, "/models"),
                ]
            else:
                urls = [
                    url_join(base, "/v1/health/ready"),
                    url_join(base, "/v1/models"),
                    url_join(base, "/health"),
                ]
            ok = check_any_url(name, [host_check_url(url) for url in urls]) and ok

    return 0 if ok or not strict else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to collection deployment YAML",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the deployment plan")
    render_env = subparsers.add_parser("render-env", help="Render Compose env file")
    render_env.add_argument("-o", "--output", type=Path, default=GENERATED_ENV)
    up = subparsers.add_parser("up", help="Start CPU-only agent services")
    up.add_argument(
        "--require-local-scene-optimizer",
        action="store_true",
        help=(
            "Fail before Docker Compose if local Scene Optimizer Core build "
            "resources cannot be fetched or validated"
        ),
    )
    subparsers.add_parser("down", help="Stop CPU-only agent services")
    subparsers.add_parser("ps", help="Show Docker Compose service state")
    subparsers.add_parser("status", help="Check configured health endpoints")
    subparsers.add_parser("smoke", help="Strict health check for agents/dependencies")
    subparsers.add_parser("brev-plan", help="Print optional Brev instance commands")
    brev_provision = subparsers.add_parser(
        "brev-provision",
        help="Create missing Brev dependency instances only with --execute",
    )
    brev_provision.add_argument(
        "--execute",
        action="store_true",
        help="Actually run brev create for missing instances",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "plan":
        return print_plan(config)
    if args.command == "render-env":
        write_env(config, args.output)
        return 0
    if args.command == "up":
        write_env(config)
        services = enabled_agent_services(config)
        if not services:
            print("No agents are enabled in the collection config")
            return 1
        resource_rc = ensure_scene_optimizer_build_resources(services)
        if resource_rc != 0:
            print(
                "WARNING: Scene Optimizer Core build-resource preflight failed "
                f"with exit code {resource_rc}; continuing because local Scene "
                "Optimizer resources are optional when remote optimization is "
                "configured or optimization is disabled."
            )
            if args.require_local_scene_optimizer:
                return resource_rc
        return run_compose(config, "up", "-d", "--build", *services)
    if args.command == "down":
        return run_compose(config, "down")
    if args.command == "ps":
        return run_compose(config, "ps")
    if args.command == "status":
        return status(config, strict=False)
    if args.command == "smoke":
        return status(config, strict=True)
    if args.command == "brev-plan":
        return print_brev_plan(config)
    if args.command == "brev-provision":
        return provision_brev(config, execute=args.execute)

    return 2


if __name__ == "__main__":
    sys.exit(main())
