#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan Brev deployments for agent services without spending credits by default."""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

ServiceName = Literal["material", "physics", "texture"]
PresetName = Literal[
    "render-only",
    "single-host-local-vlm",
    "hybrid",
    "service-only",
    "single-host-local-sidecars",
]
HybridConnectivityName = Literal[
    "auto",
    "local-port-forward",
    "external-url",
    "tunnel-url",
    "private-ip",
    "same-node",
]

QWEN_SELF_HOSTED_SMOKE_MODEL = "Qwen/Qwen3.5-4B"
TEXTURE_QWEN_SMALL_MODEL = "Qwen/Qwen3.5-4B"
COSMOS_LOCAL_VLM_MODEL = "nvidia/cosmos-reason2-8b"
TEXTURE_LOCAL_IMAGE_GEN_MODEL = "black-forest-labs/flux.2-klein-4b"
TEXTURE_LOCAL_LLM_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
BREV_REMOTE_WORKTREE_PATH = "/home/ubuntu/world-understanding"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_RSYNC_EXCLUDES = (
    ".git",
    ".venv",
    ".data",
    "docs/metrics",
    "coverage.xml",
    ".env",
    ".env.*",
    ".env-*",
    ".env *",
    ".envrc",
    "*credentials*.json",
    "*private*key*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.p8",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
)

NON_RTX_RENDER_GPU_TOKENS = (
    "A100",
    "H100",
    "H200",
    "B100",
    "B200",
    "B300",
    "V100",
)

DOCKER_DAEMON_DATA_ROOT_PYTHON = """import json
import os
from pathlib import Path

path = Path("/etc/docker/daemon.json")
text = path.read_text() if path.exists() else ""
data = json.loads(text) if text.strip() else {}
if not isinstance(data, dict):
    raise SystemExit(f"{path} must contain a JSON object")
data["data-root"] = os.environ["DOCKER_DATA_ROOT"]
path.write_text(json.dumps(data, indent=2) + "\\n")
"""

TEXTURE_CREDENTIAL_ENV_PYTHON = """import os
import sys
from pathlib import Path

from dotenv import dotenv_values

credential_names = ("NGC_API_KEY", "HF_TOKEN")
values = {name: os.environ.get(name) for name in credential_names}
dotenv_path = Path(sys.argv[1]) / ".env"
if dotenv_path.is_file():
    dotenv = dotenv_values(dotenv_path=dotenv_path, interpolate=False)
    for name in credential_names:
        # An explicit environment entry is authoritative even when invalid. Do not
        # silently replace an exported empty value with a credential from disk.
        if name not in os.environ and name in dotenv:
            values[name] = dotenv[name]

for name in credential_names:
    value = values[name]
    if not value:
        raise SystemExit(f"{name} is required")
    if "\\n" in value or "\\r" in value:
        raise SystemExit(f"{name} must be a single-line value")

sys.stdout.write(
    "".join(f"{name}={values[name]}\\n" for name in credential_names),
)
"""

REMOTE_CREDENTIAL_INSTALL_SHELL = """set -e
umask 077
destination="$HOME/.ngc-nim.env"
tmp_env="$(mktemp "${destination}.XXXXXX")"
cleanup() { rm -f -- "$tmp_env"; }
trap cleanup EXIT
trap 'cleanup; exit 1' HUP INT TERM
cat > "$tmp_env"
chmod 600 "$tmp_env"
mv -f -- "$tmp_env" "$destination"
trap - EXIT HUP INT TERM
"""

SSH_TRUST_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "UserKnownHostsFile=~/.ssh/known_hosts",
)

SSH_TRUST_OPTIONS_SHELL = (
    "-o BatchMode=yes -o ConnectTimeout=10 "
    "-o StrictHostKeyChecking=yes "
    "-o UserKnownHostsFile=~/.ssh/known_hosts"
)


@dataclass(frozen=True)
class ServiceConfig:
    name: ServiceName
    label: str
    compose_path: str
    service_port: int
    render_required: bool
    multi_gpu_path: str | None = None
    local_vlm_profile: str | None = None
    render_endpoint_var: str | None = None
    vlm_base_url_var: str | None = None
    llm_base_url_var: str | None = None
    nim_api_key_var: str | None = None
    vlm_model_var: str | None = None
    llm_model_var: str | None = None
    image_gen_base_url_var: str | None = None
    image_gen_model_var: str | None = None
    image_gen_backend_var: str | None = None
    llm_backend_var: str | None = None
    host_model_overlay_path: str | None = None


@dataclass(frozen=True)
class PlannerOptions:
    service: ServiceName
    preset: str
    name: str
    render_gpu_name: str = "RTX"
    render_min_vram_gb: int = 48
    render_min_disk_gb: int = 500
    vlm_gpu_name: str | None = None
    vlm_min_vram_gb: int | None = None
    vlm_min_disk_gb: int = 500
    cpu_type: str = "n2d-standard-4"
    qwen_model: str | None = None
    image_gen_node_name: str | None = None
    image_gen_type: str | None = "g6e.xlarge"
    image_gen_gpu_name: str = "L40S"
    image_gen_min_vram_gb: int = 48
    image_gen_min_disk_gb: int = 500
    image_gen_base_url: str | None = None
    image_gen_port: int = 8000
    image_gen_local_port: int = 8005
    texture_include_llm: bool = False
    prefer_stoppable: bool = True
    hybrid_connectivity: str = "auto"
    model_base_url: str | None = None
    model_port: int = 8000
    model_local_port: int = 8003
    render_port: int = 8001
    render_local_port: int = 8001


@dataclass(frozen=True)
class PlannedCommand:
    label: str
    argv: tuple[str, ...]
    cost_incurring: bool = False

    def shell(self) -> str:
        return shlex.join(self.argv)


@dataclass(frozen=True)
class NodePlan:
    name: str
    role: str
    purpose: str
    search: PlannedCommand | None
    dry_run: PlannedCommand
    create: PlannedCommand


@dataclass(frozen=True)
class BrevDeploymentPlan:
    service: ServiceConfig
    preset: str
    nodes: tuple[NodePlan, ...]
    environment: tuple[str, ...]
    deploy_commands: tuple[PlannedCommand, ...]
    environment_description: str = (
        "Set these on the service node before starting Compose:"
    )
    notes: tuple[str, ...] = field(default_factory=tuple)

    def render_markdown(self) -> str:
        lines = [
            f"# Brev Plan: {self.service.label} / {self.preset}",
            "",
            "This is a credit-safe plan. Commands under dry-run and inspection do not",
            "create instances. Commands under cost-incurring provisioning are for",
            "operator review before manual execution.",
            "",
            "## Credit-Safe Preflight",
            "",
        ]
        for command in preflight_commands():
            lines.extend(_command_block(command))

        lines.extend(["## Capacity Checks", ""])
        for node in self.nodes:
            lines.extend([f"### {node.name}", "", node.purpose, ""])
            if node.search:
                lines.extend(_command_block(node.search))
            lines.extend(_command_block(node.dry_run))

        lines.extend(["## Cost-Incurring Provisioning", ""])
        lines.append(
            "Do not run these until the dry-runs show acceptable capacity and price."
        )
        lines.append("")
        for node in self.nodes:
            lines.extend(_command_block(node.create))

        if self.environment:
            lines.extend(["## Endpoint Wiring", ""])
            lines.append(self.environment_description)
            lines.append("")
            lines.append("```bash")
            lines.extend(_export_environment_lines(self.environment))
            lines.append("```")
            lines.append("")

        if self.deploy_commands:
            lines.extend(["## Deploy After Provisioning", ""])
            for command in self.deploy_commands:
                lines.extend(_command_block(command))

        if self.notes:
            lines.extend(["## Notes", ""])
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"


def _export_environment_lines(environment: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for line in environment:
        key, separator, value = line.partition("=")
        if _is_environment_assignment(key, separator):
            lines.append(f"export {key}={shlex.quote(value)}")
        else:
            lines.append(line)
    return lines


def _dotenv_environment_lines(environment: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for line in environment:
        key, separator, value = line.partition("=")
        if _is_environment_assignment(key, separator):
            lines.append(f"{key}={value}")
        elif line.startswith("#"):
            lines.append(line)
    return lines


def _is_environment_assignment(key: str, separator: str) -> bool:
    return bool(
        separator
        and key
        and (key[0].isalpha() or key[0] == "_")
        and all(char.isalnum() or char == "_" for char in key)
    )


SERVICE_CONFIGS: dict[ServiceName, ServiceConfig] = {
    "material": ServiceConfig(
        name="material",
        label="Material Agent Service",
        compose_path="apps/material_agent_service/docker-compose.yml",
        multi_gpu_path="apps/material_agent_service/docker-compose.multi-gpu.yml",
        service_port=8000,
        render_required=True,
        local_vlm_profile="vlm",
        render_endpoint_var="RENDER_ENDPOINT",
        vlm_base_url_var="MA_VLM_NIM_BASE_URL",
        llm_base_url_var="MA_LLM_NIM_BASE_URL",
        nim_api_key_var="MA_NIM_API_KEY",
        vlm_model_var="MA_VLM_MODEL",
        llm_model_var="MA_LLM_MODEL",
    ),
    "physics": ServiceConfig(
        name="physics",
        label="Physics Agent Service",
        compose_path="apps/physics_agent_service/docker-compose.yml",
        multi_gpu_path="apps/physics_agent_service/docker-compose.multi-gpu.yml",
        service_port=8000,
        render_required=True,
        local_vlm_profile="vlm",
        render_endpoint_var="RENDER_ENDPOINT",
        vlm_base_url_var="PA_VLM_NIM_BASE_URL",
        llm_base_url_var="PA_LLM_NIM_BASE_URL",
        nim_api_key_var="PA_NIM_API_KEY",
        vlm_model_var="PA_VLM_MODEL",
        llm_model_var=None,
    ),
    "texture": ServiceConfig(
        name="texture",
        label="Texture Agent Service",
        compose_path="apps/texture_agent_service/docker-compose.yml",
        multi_gpu_path="apps/texture_agent_service/docker-compose.multi-gpu.yml",
        service_port=8001,
        render_required=False,
        image_gen_backend_var="TA_IMAGE_GEN_BACKEND",
        image_gen_base_url_var="TA_IMAGE_GEN_BASE_URL",
        image_gen_model_var="TA_IMAGE_GEN_MODEL",
        llm_backend_var="TA_LLM_BACKEND",
        llm_base_url_var="TA_LLM_BASE_URL",
        llm_model_var="TA_LLM_MODEL",
        host_model_overlay_path=(
            "apps/texture_agent_service/docker-compose.brev-host-llm.yml"
        ),
    ),
}

SUPPORTED_PRESETS: dict[ServiceName, tuple[str, ...]] = {
    "material": ("render-only", "single-host-local-vlm", "hybrid"),
    "physics": ("render-only", "single-host-local-vlm", "hybrid"),
    "texture": ("service-only", "single-host-local-sidecars", "hybrid"),
}


def preflight_commands() -> tuple[PlannedCommand, ...]:
    return (
        PlannedCommand("Check Brev version", ("brev", "--version")),
        PlannedCommand("Check Brev backend health", ("brev", "healthcheck")),
        PlannedCommand("List current instances", ("brev", "ls", "--json")),
    )


def build_plan(options: PlannerOptions) -> BrevDeploymentPlan:
    service = SERVICE_CONFIGS[options.service]
    if options.preset not in SUPPORTED_PRESETS[service.name]:
        supported = ", ".join(SUPPORTED_PRESETS[service.name])
        raise ValueError(
            f"Preset {options.preset!r} is not supported for {service.name}. "
            f"Supported presets: {supported}"
        )

    if options.preset == "render-only":
        return _render_only_plan(service, options)
    if options.preset == "single-host-local-vlm":
        return _single_host_local_vlm_plan(service, options)
    if options.preset == "hybrid":
        return _hybrid_plan(service, options)
    if options.preset == "service-only":
        return _texture_service_only_plan(service, options)
    if options.preset == "single-host-local-sidecars":
        return _texture_single_host_sidecars_plan(service, options)

    raise ValueError(f"Unhandled preset: {options.preset}")


SUPPORTED_HYBRID_CONNECTIVITY = (
    "auto",
    "local-port-forward",
    "external-url",
    "tunnel-url",
    "private-ip",
    "same-node",
)


def _resolve_hybrid_connectivity(
    service: ServiceConfig, options: PlannerOptions
) -> str:
    connectivity = options.hybrid_connectivity
    if connectivity not in SUPPORTED_HYBRID_CONNECTIVITY:
        supported = ", ".join(SUPPORTED_HYBRID_CONNECTIVITY)
        raise ValueError(
            f"Hybrid connectivity {connectivity!r} is not supported. "
            f"Supported values: {supported}"
        )

    if connectivity == "auto":
        return "local-port-forward"

    if connectivity == "same-node" and service.render_required:
        raise ValueError(
            "same-node hybrid connectivity is not valid for render-required "
            f"{service.name} deployments with A100/H100-grade VLM nodes. Use "
            "single-host-local-vlm for an RTX same-node deployment, or use "
            "external-url/tunnel-url for a separate high-end VLM node."
        )

    return connectivity


def _qwen_model(service: ServiceConfig, options: PlannerOptions) -> str:
    if options.qwen_model:
        return options.qwen_model
    if service.name == "texture":
        return TEXTURE_QWEN_SMALL_MODEL
    return QWEN_SELF_HOSTED_SMOKE_MODEL


def _model_gpu_name(service: ServiceConfig, options: PlannerOptions) -> str:
    if options.vlm_gpu_name:
        return options.vlm_gpu_name
    if service.name == "texture":
        return "L4"
    return "A100"


def _model_min_vram_gb(service: ServiceConfig, options: PlannerOptions) -> int:
    if options.vlm_min_vram_gb is not None:
        return options.vlm_min_vram_gb
    if service.name == "texture":
        return 24
    return 80


def _same_node_model_base_url(options: PlannerOptions) -> str:
    if options.model_base_url:
        return options.model_base_url.rstrip("/")
    return f"http://host.docker.internal:{options.model_port}/v1"


def _local_model_base_url(options: PlannerOptions) -> str:
    if options.model_base_url:
        return options.model_base_url.rstrip("/")
    return f"http://localhost:{options.model_local_port}/v1"


def _local_image_gen_base_url(options: PlannerOptions) -> str:
    if options.image_gen_base_url:
        return options.image_gen_base_url.rstrip("/")
    return f"http://localhost:{options.image_gen_local_port}/v1"


def _local_render_endpoint(options: PlannerOptions) -> str:
    return f"http://localhost:{options.render_local_port}"


def _uses_local_no_auth_endpoint(base_url: str | None) -> bool:
    if not base_url:
        return False
    parsed = urlparse(base_url)
    host = parsed.hostname
    if not host:
        return False
    host = host.lower()
    if host in {
        "localhost",
        "host.docker.internal",
        "image-gen-nim",
        "llm-nim",
        "vlm-nim",
    }:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _nim_api_key_environment(api_key_var: str, base_url: str) -> str:
    if _uses_local_no_auth_endpoint(base_url):
        return f"{api_key_var}=not-used"
    return f"# Set {api_key_var} in .env for authenticated NIM endpoints."


def _image_gen_api_key_environment(base_url: str) -> str:
    if _uses_local_no_auth_endpoint(base_url):
        return "TA_IMAGE_GEN_API_KEY=not-used"
    return "# Set TA_IMAGE_GEN_API_KEY in .env for authenticated image-generation endpoints."


def _hybrid_model_base_url(
    *,
    options: PlannerOptions,
    model_node_name: str,
    connectivity: str,
    endpoint_kind: str,
) -> str:
    if options.model_base_url:
        return options.model_base_url.rstrip("/")
    if connectivity == "external-url":
        return f"https://replace-with-exposed-{endpoint_kind}-endpoint.example/v1"
    if connectivity == "tunnel-url":
        return (
            f"https://replace-with-operator-managed-{endpoint_kind}-tunnel.example/v1"
        )
    if connectivity == "private-ip":
        return (
            f"http://replace-with-{model_node_name}-private-ip:{options.model_port}/v1"
        )
    if connectivity == "same-node":
        return _same_node_model_base_url(options)
    raise ValueError(f"Unhandled hybrid connectivity: {connectivity}")


def _connectivity_notes(connectivity: str) -> tuple[str, ...]:
    if connectivity == "local-port-forward":
        return (
            "The main pipeline runs locally and talks to Brev-hosted dependency endpoints through local port-forwards.",
            "Run each planner-generated brev port-forward command in its own terminal while the local pipeline is active.",
            "This mode does not require Brev instance-to-instance networking.",
        )
    if connectivity == "external-url":
        return (
            "Expose the model endpoint through a deliberate provider-supported URL and pass it with --model-base-url.",
            "The endpoint must be reachable from inside the service node, not just from the operator laptop.",
        )
    if connectivity == "tunnel-url":
        return (
            "Use a tunnel that gives the service node a routable URL and pass it with --model-base-url.",
            "Plain brev port-forward only reaches the operator machine; it does not connect one Brev instance to another.",
        )
    if connectivity == "private-ip":
        return (
            "Use private IP wiring only after proving reachability for the selected Brev provider/network.",
            "A prior two-node CPU test could not reach peer private IPs over ICMP or HTTP.",
        )
    return ()


def _render_only_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    _validate_render_gpu(options.render_gpu_name)
    environment = _hosted_model_environment(service, options)
    render_node = _gpu_node(
        name=f"{options.name}-{service.name}-render",
        role="RTX render/service node",
        purpose=(
            "Runs the agent service and OVRTX rendering API on an RTX-capable GPU. "
            "VLM/LLM calls use hosted provider endpoints."
        ),
        gpu_name=options.render_gpu_name,
        min_vram_gb=options.render_min_vram_gb,
        min_total_vram_gb=None,
        min_disk_gb=options.render_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(render_node,),
        environment=environment,
        deploy_commands=_compose_deploy_commands(
            service, render_node.name, environment=environment
        ),
        notes=(
            "Default hosted VLM avoids local model warmup and extra GPU credits.",
            "Prefer RTX PRO Server 6000 / AWS g7e-class GPUs for Brev OVRTX validation; use L40/L40S as lower-cost fallback candidates.",
            "A100/H100-class GPUs are intentionally not used for rendering.",
        ),
    )


def _single_host_local_vlm_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    if service.multi_gpu_path is None or service.local_vlm_profile is None:
        raise ValueError(f"{service.name} has no local VLM Compose overlay")
    _validate_render_gpu(options.render_gpu_name)
    environment = _local_vlm_overlay_environment(service)
    node = _gpu_node(
        name=f"{options.name}-{service.name}-all",
        role="multi-GPU RTX service/render/local-VLM node",
        purpose=(
            "Runs the agent service, OVRTX rendering API, and the existing local "
            "VLM NIM sidecar on separate RTX GPUs via the multi-GPU overlay."
        ),
        gpu_name=options.render_gpu_name,
        min_vram_gb=options.render_min_vram_gb,
        min_total_vram_gb=options.render_min_vram_gb * 2,
        min_disk_gb=options.render_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(node,),
        environment=environment,
        deploy_commands=_compose_deploy_commands(
            service,
            node.name,
            environment=environment,
            overlay_path=service.multi_gpu_path,
            profiles=(service.local_vlm_profile,),
        ),
        notes=(
            "This preset uses the service's existing Cosmos Reason2 8B sidecar.",
            "For Qwen 3.5/3.6-family VLMs on A100/H100-grade GPUs, use the hybrid preset.",
            "Keep GPU pinning from the multi-GPU overlay; NIM can fail if another process holds VRAM.",
        ),
    )


def _hybrid_plan(service: ServiceConfig, options: PlannerOptions) -> BrevDeploymentPlan:
    if service.name == "texture":
        return _texture_hybrid_plan(service, options)

    connectivity = _resolve_hybrid_connectivity(service, options)
    if connectivity == "local-port-forward":
        return _local_pipeline_render_vlm_hybrid_plan(service, options)

    _validate_render_gpu(options.render_gpu_name)
    render_node = _gpu_node(
        name=f"{options.name}-{service.name}-render",
        role="RTX render/service node",
        purpose=(
            "Runs the agent service and OVRTX rendering API. The service points "
            "at a separate high-end VLM node."
        ),
        gpu_name=options.render_gpu_name,
        min_vram_gb=options.render_min_vram_gb,
        min_total_vram_gb=None,
        min_disk_gb=options.render_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    vlm_node = _gpu_node(
        name=f"{options.name}-{service.name}-vlm",
        role="A100/H100-grade VLM node",
        purpose=(
            "Hosts an OpenAI-compatible VLM/LLM endpoint on A100-or-better GPU "
            "capacity. The default model wiring uses a Qwen 3.5 family model."
        ),
        gpu_name=_model_gpu_name(service, options),
        min_vram_gb=_model_min_vram_gb(service, options),
        min_total_vram_gb=None,
        min_disk_gb=options.vlm_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    model_base_url = _hybrid_model_base_url(
        options=options,
        model_node_name=vlm_node.name,
        connectivity=connectivity,
        endpoint_kind="vlm",
    )
    environment = _hybrid_vlm_environment(service, options, model_base_url)
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(render_node, vlm_node),
        environment=environment,
        deploy_commands=(
            *_model_node_qualification_commands(vlm_node.name),
            *_model_node_readiness_commands(vlm_node.name, options.model_port),
            *_compose_deploy_commands(
                service, render_node.name, environment=environment
            ),
            _service_node_model_probe_command(render_node.name, model_base_url),
        ),
        notes=(
            "Deploy the Qwen-family VLM server or Brev launchable on the VLM node before starting the service.",
            *_connectivity_notes(connectivity),
            "The render node remains RTX-only because A100/H100-class GPUs lack RTX rendering support.",
        ),
    )


def _local_pipeline_render_vlm_hybrid_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    _validate_render_gpu(options.render_gpu_name)
    render_node = _gpu_node(
        name=f"{options.name}-{service.name}-render",
        role="RTX OVRTX render endpoint node",
        purpose=(
            "Runs only the OVRTX rendering API on an RTX-capable GPU. The local "
            "pipeline reaches it through a Brev port-forward."
        ),
        gpu_name=options.render_gpu_name,
        min_vram_gb=options.render_min_vram_gb,
        min_total_vram_gb=None,
        min_disk_gb=options.render_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    vlm_node = _gpu_node(
        name=f"{options.name}-{service.name}-vlm",
        role="A100/H100-grade VLM endpoint node",
        purpose=(
            "Hosts an OpenAI-compatible Qwen-family VLM/LLM endpoint. The local "
            "pipeline reaches it through a separate Brev port-forward."
        ),
        gpu_name=_model_gpu_name(service, options),
        min_vram_gb=_model_min_vram_gb(service, options),
        min_total_vram_gb=None,
        min_disk_gb=options.vlm_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    model_base_url = _local_model_base_url(options)
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(render_node, vlm_node),
        environment=_local_pipeline_vlm_environment(
            service=service,
            options=options,
            render_endpoint=_local_render_endpoint(options),
            model_base_url=model_base_url,
        ),
        environment_description=(
            "Set these on the local machine before running the agent pipeline:"
        ),
        deploy_commands=(
            *_standalone_ovrtx_deploy_commands(
                render_node.name,
                options.render_local_port,
                options.render_port,
            ),
            *_model_node_qualification_commands(vlm_node.name),
            *_model_node_readiness_commands(vlm_node.name, options.model_port),
            _port_forward_command(
                "Forward the VLM/LLM endpoint to localhost",
                vlm_node.name,
                options.model_local_port,
                options.model_port,
            ),
            _local_probe_command("Verify local VLM/LLM endpoint", model_base_url),
        ),
        notes=(
            "Deploy the Qwen-family VLM server or Brev launchable on the VLM node before starting the local pipeline.",
            *_connectivity_notes("local-port-forward"),
            "The render node remains RTX-only because A100/H100-class GPUs lack RTX rendering support.",
        ),
    )


def _texture_service_only_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    environment = _texture_service_only_environment(options)
    node = _cpu_node(
        name=f"{options.name}-texture-service",
        role="CPU texture service node",
        purpose=(
            "Runs texture-agent-service with hosted image generation and hosted LLM "
            "providers. Use this as a secondary service-on-Brev validation path."
        ),
        cpu_type=options.cpu_type,
    )
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(node,),
        environment=environment,
        deploy_commands=_compose_deploy_commands(
            service, node.name, environment=environment
        ),
        notes=(
            "Texture service does not need OVRTX rendering.",
            "The target workflow for this worktree is local pipeline plus Brev-hosted dependency endpoints.",
            "For Docker Compose service modes, put real TA_IMAGE_GEN_API_KEY, TA_LLM_API_KEY, or TA_NIM_API_KEY values in the repo-root or service .env file; plain local exports only affect local CLI pipelines.",
        ),
    )


def _texture_single_host_sidecars_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    if service.multi_gpu_path is None:
        raise ValueError("Texture service has no local sidecar Compose overlay")
    environment = (
        "TA_IMAGE_GEN_BACKEND=openai",
        f"TA_IMAGE_GEN_MODEL={TEXTURE_LOCAL_IMAGE_GEN_MODEL}",
        "TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1",
        "TA_IMAGE_GEN_API_KEY=not-used",
        "TA_LLM_BACKEND=nim",
        f"TA_LLM_MODEL={TEXTURE_LOCAL_LLM_MODEL}",
        "TA_LLM_BASE_URL=http://llm-nim:8000/v1",
        "TA_NIM_API_KEY=not-used",
    )
    node = _gpu_node(
        name=f"{options.name}-texture-sidecars",
        role="multi-GPU texture service/local-sidecar node",
        purpose=(
            "Runs texture-agent-service with local image-gen and LLM NIM sidecars "
            "pinned to separate GPUs via the multi-GPU overlay."
        ),
        gpu_name=options.render_gpu_name,
        min_vram_gb=options.render_min_vram_gb,
        min_total_vram_gb=options.render_min_vram_gb * 2,
        min_disk_gb=options.render_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(node,),
        environment=environment,
        deploy_commands=_compose_deploy_commands(
            service,
            node.name,
            environment=environment,
            overlay_path=service.multi_gpu_path,
            profiles=("image-gen", "llm"),
        ),
        notes=(
            "This preset uses the service's existing FLUX image-gen sidecar and Nemotron Nano LLM sidecar.",
            "Use the texture hybrid preset for a separate small Qwen LLM endpoint.",
        ),
    )


def _texture_hybrid_plan(
    service: ServiceConfig, options: PlannerOptions
) -> BrevDeploymentPlan:
    connectivity = _resolve_hybrid_connectivity(service, options)
    if connectivity == "local-port-forward":
        image_gen_nodes: tuple[NodePlan, ...] = ()
        image_gen_commands: tuple[PlannedCommand, ...] = ()
        if not options.image_gen_base_url:
            image_gen_node_name = (
                options.image_gen_node_name or f"{options.name}-texture-image-gen"
            )
            image_gen_purpose = (
                "Hosts an OpenAI-compatible FLUX image-generation endpoint. "
                "The local texture pipeline reaches it through a Brev "
                "port-forward."
            )
            if options.image_gen_type:
                image_gen_node = _gpu_type_node(
                    name=image_gen_node_name,
                    role="texture image-generation endpoint node",
                    purpose=(
                        f"{image_gen_purpose} Defaults to the validated AWS "
                        f"{options.image_gen_type} L40S path."
                    ),
                    instance_type=options.image_gen_type,
                    min_disk_gb=options.image_gen_min_disk_gb,
                )
            else:
                image_gen_node = _gpu_node(
                    name=image_gen_node_name,
                    role="texture image-generation endpoint node",
                    purpose=image_gen_purpose,
                    gpu_name=options.image_gen_gpu_name,
                    min_vram_gb=options.image_gen_min_vram_gb,
                    min_total_vram_gb=None,
                    min_disk_gb=options.image_gen_min_disk_gb,
                    prefer_stoppable=options.prefer_stoppable,
                )
            image_gen_nodes = (image_gen_node,)
            image_gen_commands = (
                *_texture_image_gen_setup_commands(
                    image_gen_node.name, options.image_gen_port
                ),
                _image_gen_node_readiness_command(
                    image_gen_node.name, options.image_gen_port
                ),
                _port_forward_command(
                    "Forward the image-generation endpoint to localhost",
                    image_gen_node.name,
                    options.image_gen_local_port,
                    options.image_gen_port,
                ),
            )
        image_gen_base_url = _local_image_gen_base_url(options)
        llm_nodes: tuple[NodePlan, ...] = ()
        llm_commands: tuple[PlannedCommand, ...] = ()
        model_base_url: str | None = None
        if options.texture_include_llm:
            llm_node = _gpu_node(
                name=f"{options.name}-texture-llm",
                role="texture LLM endpoint node",
                purpose=(
                    "Hosts an OpenAI-compatible Qwen-family LLM endpoint for "
                    "auto-prompt generation. Skip this node when the local texture "
                    "config provides explicit prompts for every material."
                ),
                gpu_name=_model_gpu_name(service, options),
                min_vram_gb=_model_min_vram_gb(service, options),
                min_total_vram_gb=None,
                min_disk_gb=options.vlm_min_disk_gb,
                prefer_stoppable=options.prefer_stoppable,
            )
            model_base_url = _local_model_base_url(options)
            llm_nodes = (llm_node,)
            llm_commands = (
                *_model_node_qualification_commands(llm_node.name),
                *_model_node_readiness_commands(llm_node.name, options.model_port),
                _port_forward_command(
                    "Forward the LLM endpoint to localhost",
                    llm_node.name,
                    options.model_local_port,
                    options.model_port,
                ),
                _local_probe_command("Verify local LLM endpoint", model_base_url),
            )
        return BrevDeploymentPlan(
            service=service,
            preset=options.preset,
            nodes=(*image_gen_nodes, *llm_nodes),
            environment=_texture_hybrid_environment(
                options=options,
                image_gen_base_url=image_gen_base_url,
                llm_base_url=model_base_url,
            ),
            environment_description=(
                "Set these on the local machine before running the texture pipeline:"
            ),
            deploy_commands=(
                *image_gen_commands,
                _local_probe_command(
                    "Verify image-generation endpoint",
                    f"{image_gen_base_url}/health/ready",
                ),
                *llm_commands,
            ),
            notes=(
                "Texture has no rendering requirement; the main texture pipeline stays local and uses Brev-hosted dependency endpoints.",
                "The image-generation endpoint is the required heavy dependency for generate_textures.",
                "The LLM endpoint is optional and skipped by default when explicit material prompts are supplied in the config.",
                "Pass --texture-include-llm to add a small Qwen LLM node for auto-prompt generation.",
                "Before direct SSH credential transfer, install the node host key in ~/.ssh/known_hosts only after verifying its fingerprint through an independently authenticated Brev/provider channel; generated SSH commands enforce strict verification.",
                *_connectivity_notes("local-port-forward"),
            ),
        )

    if connectivity == "same-node":
        if service.host_model_overlay_path is None:
            raise ValueError(f"{service.name} has no host-model Compose overlay")
        node = _gpu_node(
            name=f"{options.name}-texture-all",
            role="texture service/LLM node",
            purpose=(
                "Runs texture-agent-service on the same Brev node as an "
                "operator-managed Qwen-family LLM endpoint. The service "
                "container reaches the host endpoint through host.docker.internal."
            ),
            gpu_name=_model_gpu_name(service, options),
            min_vram_gb=_model_min_vram_gb(service, options),
            min_total_vram_gb=None,
            min_disk_gb=options.vlm_min_disk_gb,
            prefer_stoppable=options.prefer_stoppable,
        )
        model_base_url = _same_node_model_base_url(options)
        environment = _texture_hybrid_environment(
            options=options,
            image_gen_base_url=options.image_gen_base_url,
            llm_base_url=model_base_url,
        )
        return BrevDeploymentPlan(
            service=service,
            preset=options.preset,
            nodes=(node,),
            environment=environment,
            deploy_commands=(
                *_model_node_qualification_commands(node.name),
                *_model_node_readiness_commands(node.name, options.model_port),
                *_compose_deploy_commands(
                    service,
                    node.name,
                    environment=environment,
                    overlay_path=service.host_model_overlay_path,
                ),
                _service_container_model_probe_command(
                    node.name,
                    "texture-agent-service",
                    model_base_url,
                ),
            ),
            notes=(
                "This is a secondary service-on-Brev validation path; the target workflow keeps the texture pipeline local.",
                "Use a smaller Qwen 3.5-family model for texture by default.",
                "Start the Qwen-family OpenAI-compatible endpoint on the Brev host before starting Compose.",
                "The host-model overlay adds host.docker.internal so the service container can reach the host endpoint.",
                "Set TA_IMAGE_GEN_BASE_URL when routing the service at a dedicated image-generation endpoint.",
                "For Docker Compose service modes, put real TA_IMAGE_GEN_API_KEY, TA_LLM_API_KEY, or TA_NIM_API_KEY values in the repo-root or service .env file; plain local exports only affect local CLI pipelines.",
            ),
        )

    control_node = _cpu_node(
        name=f"{options.name}-texture-service",
        role="CPU texture service node",
        purpose="Runs texture-agent-service and points at a separate LLM endpoint.",
        cpu_type=options.cpu_type,
    )
    llm_node = _gpu_node(
        name=f"{options.name}-texture-llm",
        role="texture LLM node",
        purpose=(
            "Hosts an OpenAI-compatible Qwen 3.5/3.6-family LLM endpoint for "
            "texture prompt generation."
        ),
        gpu_name=_model_gpu_name(service, options),
        min_vram_gb=_model_min_vram_gb(service, options),
        min_total_vram_gb=None,
        min_disk_gb=options.vlm_min_disk_gb,
        prefer_stoppable=options.prefer_stoppable,
    )
    model_base_url = _hybrid_model_base_url(
        options=options,
        model_node_name=llm_node.name,
        connectivity=connectivity,
        endpoint_kind="llm",
    )
    environment = _texture_hybrid_environment(
        options=options,
        image_gen_base_url=options.image_gen_base_url,
        llm_base_url=model_base_url,
    )
    return BrevDeploymentPlan(
        service=service,
        preset=options.preset,
        nodes=(control_node, llm_node),
        environment=environment,
        deploy_commands=(
            *_model_node_qualification_commands(llm_node.name),
            *_model_node_readiness_commands(llm_node.name, options.model_port),
            *_compose_deploy_commands(
                service, control_node.name, environment=environment
            ),
            _service_node_model_probe_command(control_node.name, model_base_url),
        ),
        notes=(
            "Texture has no VLM/rendering requirement; use small Qwen 3.5 by default and scale the GPU only when needed.",
            *_connectivity_notes(connectivity),
            "Set TA_IMAGE_GEN_BASE_URL when routing the service at a dedicated image-generation endpoint.",
        ),
    )


def _gpu_node(
    *,
    name: str,
    role: str,
    purpose: str,
    gpu_name: str,
    min_vram_gb: int,
    min_total_vram_gb: int | None,
    min_disk_gb: int,
    prefer_stoppable: bool,
) -> NodePlan:
    search_argv = [
        "brev",
        "search",
        "gpu",
        "--gpu-name",
        gpu_name,
        "--min-vram",
        str(min_vram_gb),
        "--min-disk",
        str(min_disk_gb),
        "--sort",
        "price",
        "--json",
    ]
    create_argv = [
        "brev",
        "create",
        name,
        "--gpu-name",
        gpu_name,
        "--min-vram",
        str(min_vram_gb),
        "--min-disk",
        str(min_disk_gb),
        "--sort",
        "price",
    ]
    if min_total_vram_gb is not None:
        search_argv.extend(["--min-total-vram", str(min_total_vram_gb)])
        create_argv.extend(["--min-total-vram", str(min_total_vram_gb)])
    if prefer_stoppable:
        search_argv.append("--stoppable")
        create_argv.append("--stoppable")

    return NodePlan(
        name=name,
        role=role,
        purpose=purpose,
        search=PlannedCommand(f"Search {role} candidates", tuple(search_argv)),
        dry_run=PlannedCommand(
            f"Dry-run {role} provisioning",
            (*create_argv[:3], "--dry-run", *create_argv[3:]),
        ),
        create=PlannedCommand(
            f"Create {role}",
            tuple(create_argv),
            cost_incurring=True,
        ),
    )


def _gpu_type_node(
    *,
    name: str,
    role: str,
    purpose: str,
    instance_type: str,
    min_disk_gb: int,
) -> NodePlan:
    create_argv = [
        "brev",
        "create",
        name,
        "--type",
        instance_type,
        "--min-disk",
        str(min_disk_gb),
    ]
    return NodePlan(
        name=name,
        role=role,
        purpose=purpose,
        search=None,
        dry_run=PlannedCommand(
            f"Dry-run {role} provisioning",
            (*create_argv[:3], "--dry-run", *create_argv[3:]),
        ),
        create=PlannedCommand(
            f"Create {role}",
            tuple(create_argv),
            cost_incurring=True,
        ),
    )


def _cpu_node(*, name: str, role: str, purpose: str, cpu_type: str) -> NodePlan:
    return NodePlan(
        name=name,
        role=role,
        purpose=purpose,
        search=PlannedCommand(
            "Search CPU candidates",
            ("brev", "search", "cpu", "--sort", "price", "--json"),
        ),
        dry_run=PlannedCommand(
            f"Dry-run {role} provisioning",
            ("brev", "create", name, "--dry-run", "--type", cpu_type),
        ),
        create=PlannedCommand(
            f"Create {role}",
            ("brev", "create", name, "--type", cpu_type),
            cost_incurring=True,
        ),
    )


def _hosted_model_environment(
    service: ServiceConfig, options: PlannerOptions
) -> tuple[str, ...]:
    if service.name == "texture":
        return ()

    env = [
        f"{service.render_endpoint_var}=http://ovrtx-rendering-api:8000",
    ]
    if options.qwen_model:
        env.append(f"{service.vlm_model_var}={options.qwen_model}")
        if service.llm_model_var:
            env.append(f"{service.llm_model_var}={options.qwen_model}")
    else:
        env.append("# Model env vars omitted; service defaults select hosted models.")
    env.append("# Set NVIDIA_API_KEY or another supported hosted-provider key in .env.")
    return tuple(env)


def _local_vlm_overlay_environment(service: ServiceConfig) -> tuple[str, ...]:
    vlm_base_url = "http://vlm-nim:8000/v1"
    env = [
        f"{service.render_endpoint_var}=http://ovrtx-rendering-api:8000",
        f"{service.vlm_base_url_var}={vlm_base_url}",
        f"{service.vlm_model_var}={COSMOS_LOCAL_VLM_MODEL}",
    ]
    if service.llm_base_url_var:
        env.append(f"{service.llm_base_url_var}={vlm_base_url}")
    if service.llm_model_var:
        env.append(f"{service.llm_model_var}={COSMOS_LOCAL_VLM_MODEL}")
    if service.nim_api_key_var:
        env.append(_nim_api_key_environment(service.nim_api_key_var, vlm_base_url))
    return tuple(env)


def _hybrid_vlm_environment(
    service: ServiceConfig, options: PlannerOptions, model_base_url: str
) -> tuple[str, ...]:
    env = [
        f"{service.render_endpoint_var}=http://ovrtx-rendering-api:8000",
        f"{service.vlm_base_url_var}={model_base_url}",
        f"{service.vlm_model_var}={_qwen_model(service, options)}",
    ]
    if service.llm_base_url_var:
        env.append(f"{service.llm_base_url_var}={model_base_url}")
    if service.llm_model_var:
        env.append(f"{service.llm_model_var}={_qwen_model(service, options)}")
    if service.nim_api_key_var:
        env.append(_nim_api_key_environment(service.nim_api_key_var, model_base_url))
    return tuple(env)


def _local_pipeline_vlm_environment(
    *,
    service: ServiceConfig,
    options: PlannerOptions,
    render_endpoint: str,
    model_base_url: str,
) -> tuple[str, ...]:
    env = [
        f"{service.render_endpoint_var}={render_endpoint}",
        f"{service.vlm_base_url_var}={model_base_url}",
        f"{service.vlm_model_var}={_qwen_model(service, options)}",
    ]
    if service.render_required:
        env.insert(1, "MA_RENDERING_USE_DATA_URI=true")
    if service.llm_base_url_var:
        env.append(f"{service.llm_base_url_var}={model_base_url}")
    if service.llm_model_var:
        env.append(f"{service.llm_model_var}={_qwen_model(service, options)}")
    if service.nim_api_key_var:
        env.append(_nim_api_key_environment(service.nim_api_key_var, model_base_url))
    return tuple(env)


def _texture_hybrid_environment(
    *,
    options: PlannerOptions,
    image_gen_base_url: str | None,
    llm_base_url: str | None,
) -> tuple[str, ...]:
    env = []
    if image_gen_base_url:
        env.extend(
            [
                "TA_IMAGE_GEN_BACKEND=openai",
                f"TA_IMAGE_GEN_BASE_URL={image_gen_base_url}",
                f"TA_IMAGE_GEN_MODEL={TEXTURE_LOCAL_IMAGE_GEN_MODEL}",
            ]
        )
        env.append(_image_gen_api_key_environment(image_gen_base_url))
    else:
        env.extend(
            [
                "TA_IMAGE_GEN_BACKEND=nim",
                "# Optionally set TA_IMAGE_GEN_BASE_URL to a dedicated local image-gen endpoint.",
            ]
        )
    if llm_base_url:
        env.extend(
            [
                "TA_LLM_BACKEND=nim",
                f"TA_LLM_BASE_URL={llm_base_url}",
                f"TA_LLM_MODEL={_qwen_model(SERVICE_CONFIGS['texture'], options)}",
                _nim_api_key_environment("TA_NIM_API_KEY", llm_base_url),
            ]
        )
    else:
        env.append("# TA_LLM_* omitted; explicit prompts should cover every material.")
    return tuple(env)


def _texture_service_only_environment(options: PlannerOptions) -> tuple[str, ...]:
    env = [
        "TA_IMAGE_GEN_BACKEND=nim",
    ]
    if options.qwen_model:
        env.append(f"TA_LLM_MODEL={options.qwen_model}")
    else:
        env.append(
            "# LLM model env vars omitted; service defaults select hosted models."
        )
    env.append(
        "# Set NVIDIA_API_KEY, OPENAI_API_KEY, or another supported hosted-provider key in .env."
    )
    return tuple(env)


def _compose_deploy_commands(
    service: ServiceConfig,
    node_name: str,
    *,
    environment: tuple[str, ...] = (),
    overlay_path: str | None = None,
    profiles: tuple[str, ...] = (),
) -> tuple[PlannedCommand, ...]:
    compose_parts = ["docker", "compose"]
    if _dotenv_environment_lines(environment):
        compose_parts.extend(["--env-file", ".env"])
    compose_parts.extend(["-f", service.compose_path])
    if overlay_path:
        compose_parts.extend(["-f", overlay_path])
    for profile in profiles:
        compose_parts.extend(["--profile", profile])
    compose_parts.extend(["up", "-d", "--build"])
    compose_cmd = " ".join(shlex.quote(part) for part in compose_parts)
    if service.render_required:
        compose_cmd = f"OVRTX_RENDER_MODE=pt {compose_cmd}"

    return (
        _copy_worktree_command(
            "Copy the current worktree to the provisioned node", node_name
        ),
        *_remote_env_file_commands(node_name, environment),
        PlannedCommand(
            "Start the service stack on the provisioned node",
            (
                "brev",
                "exec",
                node_name,
                f"cd {BREV_REMOTE_WORKTREE_PATH} && {compose_cmd}",
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Forward the service API to localhost",
            (
                "brev",
                "port-forward",
                node_name,
                "-p",
                f"{service.service_port}:{service.service_port}",
            ),
            cost_incurring=True,
        ),
    )


def _remote_env_file_commands(
    node_name: str, environment: tuple[str, ...]
) -> tuple[PlannedCommand, ...]:
    lines = _dotenv_environment_lines(environment)
    if not lines:
        return ()
    quoted_lines = " ".join(shlex.quote(line) for line in lines)
    return (
        PlannedCommand(
            "Write planned endpoint env file on the provisioned node",
            (
                "brev",
                "exec",
                node_name,
                f"cd {BREV_REMOTE_WORKTREE_PATH} && "
                f"umask 077 && printf '%s\\n' {quoted_lines} > .env",
            ),
            cost_incurring=True,
        ),
    )


def _copy_worktree_command(
    label: str, node_name: str, extra_excludes: tuple[str, ...] = ()
) -> PlannedCommand:
    argv = ["rsync", "-az", "--delete"]
    for exclude in (*WORKTREE_RSYNC_EXCLUDES, *extra_excludes):
        argv.extend(["--exclude", exclude])
    argv.extend(["./", f"{node_name}:{BREV_REMOTE_WORKTREE_PATH}/"])
    return PlannedCommand(label, tuple(argv), cost_incurring=True)


def _standalone_ovrtx_deploy_commands(
    node_name: str, local_port: int, remote_port: int
) -> tuple[PlannedCommand, ...]:
    compose_path = "apps/ovrtx_rendering_api/docker-compose.yml"
    compose_cmd = (
        f"OVRTX_HOST_PORT={remote_port} OVRTX_RENDER_MODE=pt "
        f"docker compose -f {shlex.quote(compose_path)} "
        "up -d --build ovrtx-rendering-api"
    )
    return (
        _copy_worktree_command(
            "Copy the current worktree to the render node",
            node_name,
            extra_excludes=(".build-resources",),
        ),
        PlannedCommand(
            "Create empty build-resources directory on the render node",
            (
                "brev",
                "exec",
                node_name,
                f"mkdir -p {BREV_REMOTE_WORKTREE_PATH}/.build-resources",
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Start standalone OVRTX rendering API on the render node",
            (
                "brev",
                "exec",
                node_name,
                f"cd {BREV_REMOTE_WORKTREE_PATH} && {compose_cmd}",
            ),
            cost_incurring=True,
        ),
        _port_forward_command(
            "Forward the OVRTX render endpoint to localhost",
            node_name,
            local_port,
            remote_port,
        ),
        _local_probe_command(
            "Verify local OVRTX render endpoint",
            f"http://localhost:{local_port}/health",
        ),
    )


def _port_forward_command(
    label: str, node_name: str, local_port: int, remote_port: int
) -> PlannedCommand:
    return PlannedCommand(
        label,
        (
            "brev",
            "port-forward",
            node_name,
            "-p",
            f"{local_port}:{remote_port}",
        ),
        cost_incurring=True,
    )


def _local_probe_command(label: str, base_url_or_url: str) -> PlannedCommand:
    url = base_url_or_url
    if url.rstrip("/").endswith("/v1"):
        url = _models_url(url)
    return PlannedCommand(label, ("curl", "-fsS", url))


def _model_node_readiness_commands(
    node_name: str, model_port: int
) -> tuple[PlannedCommand, ...]:
    return (
        PlannedCommand(
            "Verify the model endpoint is ready on the model node",
            (
                "brev",
                "exec",
                node_name,
                f"curl -fsS http://localhost:{model_port}/v1/models",
            ),
            cost_incurring=True,
        ),
    )


def _texture_image_gen_setup_commands(
    node_name: str, image_gen_port: int
) -> tuple[PlannedCommand, ...]:
    _require_texture_credential_parser()
    return (
        PlannedCommand(
            "Qualify texture image-gen node disk, Docker storage, and GPU",
            (
                "brev",
                "exec",
                node_name,
                "df -h / /home /opt/dlami/nvme /mnt/* 2>/dev/null || true; "
                "lsblk -f; "
                "docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; "
                "nvidia-smi --query-gpu=name,memory.total,driver_version "
                "--format=csv,noheader",
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Move Docker and FLUX NIM cache to the AWS ephemeral disk",
            (
                "brev",
                "exec",
                node_name,
                "sudo mkdir -p /opt/dlami/nvme/docker "
                "/opt/dlami/nvme/nim-cache && "
                "sudo chown -R $(id -un):$(id -gn) "
                "/opt/dlami/nvme/nim-cache && "
                f"{_docker_daemon_data_root_command('/opt/dlami/nvme/docker')} && "
                "sudo nvidia-ctk runtime configure "
                "--runtime=docker --set-as-default && "
                "sudo systemctl restart docker && "
                "sudo chmod -R u+rwX,g+rwX,o-rwx "
                "/opt/dlami/nvme/nim-cache && "
                "docker info --format 'Docker Root Dir: {{.DockerRootDir}}' && "
                "df -h / /opt/dlami/nvme",
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Verify direct SSH access to the texture image-gen node",
            (
                "ssh",
                *SSH_TRUST_OPTIONS,
                node_name,
                "true",
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Stream a minimal NGC/HF env file securely to FLUX NIM",
            (
                "bash",
                "-lc",
                "set -e -o pipefail; "
                'credential_env="$("$2" -c "$3" "$5")"; '
                'printf "%s\\n" "$credential_env" | '
                f'ssh {SSH_TRUST_OPTIONS_SHELL} "$1" "$4"',
                "bash",
                node_name,
                sys.executable,
                TEXTURE_CREDENTIAL_ENV_PYTHON,
                REMOTE_CREDENTIAL_INSTALL_SHELL,
                str(REPO_ROOT),
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Log Docker into nvcr.io without exposing the NGC token",
            (
                "ssh",
                *SSH_TRUST_OPTIONS,
                node_name,
                'chmod 600 "$HOME/.ngc-nim.env" && '
                'NGC_API_KEY="$(sed -n '
                '\'s/^NGC_API_KEY=//p\' "$HOME/.ngc-nim.env")" && '
                'test -n "$NGC_API_KEY" && '
                'cleaned="$(printf %s "$NGC_API_KEY" | tr -d \'\\r\\n\')" && '
                'test "$cleaned" = "$NGC_API_KEY" && '
                "printf '%s\\n' \"$NGC_API_KEY\" | "
                'docker login nvcr.io -u "\\$oauthtoken" --password-stdin',
            ),
            cost_incurring=True,
        ),
        PlannedCommand(
            "Start the FLUX image-generation NIM",
            (
                "ssh",
                *SSH_TRUST_OPTIONS,
                node_name,
                "docker rm -f flux-image-gen >/dev/null 2>&1 || true; "
                "docker run -d --name flux-image-gen --gpus all --shm-size=32g "
                f"-p {image_gen_port}:8000 "
                '--env-file "$HOME/.ngc-nim.env" '
                "-e NIM_CACHE_PATH=/opt/nim/.cache "
                "-v /opt/dlami/nvme/nim-cache:/opt/nim/.cache "
                "nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant",
            ),
            cost_incurring=True,
        ),
    )


def _require_texture_credential_parser() -> None:
    if importlib.util.find_spec("dotenv") is None:
        raise RuntimeError(
            "python-dotenv is required to render the FLUX credential transfer. "
            "Activate the repository .venv or install the development dependencies "
            "before rendering this plan."
        )


def _image_gen_node_readiness_command(
    node_name: str, image_gen_port: int
) -> PlannedCommand:
    return PlannedCommand(
        "Verify the image-generation endpoint is ready on the image-gen node",
        (
            "brev",
            "exec",
            node_name,
            f"curl -fsS http://localhost:{image_gen_port}/v1/health/ready",
        ),
        cost_incurring=True,
    )


def _docker_daemon_data_root_command(data_root: str) -> str:
    return (
        f"sudo env DOCKER_DATA_ROOT={shlex.quote(data_root)} "
        f"python3 -c {shlex.quote(DOCKER_DAEMON_DATA_ROOT_PYTHON)}"
    )


def _model_node_qualification_commands(node_name: str) -> tuple[PlannedCommand, ...]:
    return (
        PlannedCommand(
            "Qualify model node disk, Docker storage, and GPU before pulling models",
            (
                "brev",
                "exec",
                node_name,
                "df -h / /home /mnt/* 2>/dev/null || true; "
                "lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; "
                "docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; "
                "du -sh /var/lib/docker /home/ubuntu/.cache/huggingface "
                "2>/dev/null || true; "
                "nvidia-smi --query-gpu=name,memory.total,driver_version "
                "--format=csv,noheader",
            ),
            cost_incurring=True,
        ),
    )


def _service_node_model_probe_command(
    node_name: str, model_base_url: str
) -> PlannedCommand:
    return PlannedCommand(
        "Verify model endpoint reachability from the service node",
        (
            "brev",
            "exec",
            node_name,
            f"curl -fsS {shlex.quote(_models_url(model_base_url))}",
        ),
        cost_incurring=True,
    )


def _service_container_model_probe_command(
    node_name: str, container_name: str, model_base_url: str
) -> PlannedCommand:
    return PlannedCommand(
        "Verify model endpoint reachability from the service container",
        (
            "brev",
            "exec",
            node_name,
            (
                f"docker exec {container_name} "
                f"curl -fsS {shlex.quote(_models_url(model_base_url))}"
            ),
        ),
        cost_incurring=True,
    )


def _models_url(model_base_url: str) -> str:
    return f"{model_base_url.rstrip('/')}/models"


def _validate_render_gpu(gpu_name: str) -> None:
    normalized = gpu_name.upper().replace(" ", "")
    if any(token in normalized for token in NON_RTX_RENDER_GPU_TOKENS):
        raise ValueError(
            f"{gpu_name!r} is not valid for an OVRTX render node. "
            "Use RTX-capable GPUs such as RTX PRO Server 6000, L40, L40S, A6000, or RTX6000 Ada."
        )


def _command_block(command: PlannedCommand) -> list[str]:
    marker = " # cost-incurring" if command.cost_incurring else ""
    return [f"**{command.label}**{marker}", "", "```bash", command.shell(), "```", ""]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a credit-safe Brev deployment plan for material, physics, "
            "or texture agent services."
        )
    )
    parser.add_argument("--service", choices=sorted(SERVICE_CONFIGS), required=True)
    parser.add_argument(
        "--preset",
        required=True,
        help=(
            "Preset name. material/physics: render-only, single-host-local-vlm, "
            "hybrid. texture: service-only, single-host-local-sidecars, hybrid."
        ),
    )
    parser.add_argument("--name", default="wu", help="Brev instance name prefix")
    parser.add_argument(
        "--render-gpu-name",
        default="RTX",
        help=(
            "OVRTX render-node GPU filter. For texture single-host-local-sidecars, "
            "this is the shared model-sidecar GPU filter and does not require RTX."
        ),
    )
    parser.add_argument("--render-min-vram-gb", type=int, default=48)
    parser.add_argument("--render-min-disk-gb", type=int, default=500)
    parser.add_argument(
        "--vlm-gpu-name",
        help="Model endpoint GPU filter. Defaults to A100 for material/physics and L4 for texture.",
    )
    parser.add_argument(
        "--vlm-min-vram-gb",
        type=int,
        help="Minimum VRAM for model endpoint nodes. Defaults to 80 for material/physics and 24 for texture.",
    )
    parser.add_argument("--vlm-min-disk-gb", type=int, default=500)
    parser.add_argument("--cpu-type", default="n2d-standard-4")
    parser.add_argument(
        "--qwen-model",
        help=(
            "Override the Qwen model. Defaults to the validated small Qwen 3.5 "
            "smoke model for material/physics and texture."
        ),
    )
    parser.add_argument(
        "--image-gen-node-name",
        help=(
            "Exact Brev node name for the texture image-generation endpoint. "
            "Defaults to <name>-texture-image-gen."
        ),
    )
    parser.add_argument(
        "--image-gen-type",
        default="g6e.xlarge",
        help=(
            "Exact Brev instance type for the texture image-generation endpoint. "
            "Defaults to the validated AWS g6e.xlarge L40S path. Pass an empty "
            "string to use --image-gen-gpu-name search filters instead."
        ),
    )
    parser.add_argument(
        "--image-gen-gpu-name",
        default="L40S",
        help=(
            "Texture image-generation endpoint GPU filter used when "
            "--image-gen-type is empty. Defaults to L40S."
        ),
    )
    parser.add_argument(
        "--image-gen-min-vram-gb",
        type=int,
        default=48,
        help="Minimum VRAM for texture image-generation endpoint nodes.",
    )
    parser.add_argument(
        "--image-gen-min-disk-gb",
        type=int,
        default=500,
        help="Minimum disk for texture image-generation endpoint nodes.",
    )
    parser.add_argument(
        "--image-gen-base-url",
        help="Override texture image-generation base URL instead of localhost port-forward wiring.",
    )
    parser.add_argument(
        "--image-gen-port",
        type=int,
        default=8000,
        help="Remote texture image-generation endpoint port.",
    )
    parser.add_argument(
        "--image-gen-local-port",
        type=int,
        default=8005,
        help="Local port for texture image-generation endpoint port-forward.",
    )
    parser.add_argument(
        "--texture-include-llm",
        action="store_true",
        help=(
            "For texture hybrid local-port-forward plans, add an optional small "
            "Qwen LLM node for auto-prompt generation. By default texture hybrid "
            "plans only provision image generation and assume explicit prompts."
        ),
    )
    parser.add_argument(
        "--hybrid-connectivity",
        choices=SUPPORTED_HYBRID_CONNECTIVITY,
        default="auto",
        help=(
            "Connectivity for hybrid presets. auto uses local-port-forward so "
            "the pipeline runs locally and reaches Brev dependency endpoints."
        ),
    )
    parser.add_argument(
        "--model-base-url",
        help=(
            "OpenAI-compatible /v1 base URL for a hybrid VLM/LLM endpoint. "
            "Use this with external-url or tunnel-url once the endpoint exists."
        ),
    )
    parser.add_argument(
        "--model-port",
        type=int,
        default=8000,
        help="Host port where the hybrid model endpoint listens before /v1.",
    )
    parser.add_argument(
        "--model-local-port",
        type=int,
        default=8003,
        help="Local port used when forwarding the hybrid model endpoint.",
    )
    parser.add_argument(
        "--render-port",
        type=int,
        default=8001,
        help="Remote host port where standalone OVRTX is published.",
    )
    parser.add_argument(
        "--render-local-port",
        type=int,
        default=8001,
        help="Local port used when forwarding the OVRTX render endpoint.",
    )
    parser.add_argument(
        "--allow-non-stoppable",
        action="store_true",
        help="Do not add --stoppable to Brev GPU searches/dry-runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    options = PlannerOptions(
        service=args.service,
        preset=args.preset,
        name=args.name,
        render_gpu_name=args.render_gpu_name,
        render_min_vram_gb=args.render_min_vram_gb,
        render_min_disk_gb=args.render_min_disk_gb,
        vlm_gpu_name=args.vlm_gpu_name,
        vlm_min_vram_gb=args.vlm_min_vram_gb,
        vlm_min_disk_gb=args.vlm_min_disk_gb,
        cpu_type=args.cpu_type,
        qwen_model=args.qwen_model,
        image_gen_node_name=args.image_gen_node_name,
        image_gen_type=args.image_gen_type or None,
        image_gen_gpu_name=args.image_gen_gpu_name,
        image_gen_min_vram_gb=args.image_gen_min_vram_gb,
        image_gen_min_disk_gb=args.image_gen_min_disk_gb,
        image_gen_base_url=args.image_gen_base_url,
        image_gen_port=args.image_gen_port,
        image_gen_local_port=args.image_gen_local_port,
        texture_include_llm=args.texture_include_llm,
        prefer_stoppable=not args.allow_non_stoppable,
        hybrid_connectivity=args.hybrid_connectivity,
        model_base_url=args.model_base_url,
        model_port=args.model_port,
        model_local_port=args.model_local_port,
        render_port=args.render_port,
        render_local_port=args.render_local_port,
    )
    print(build_plan(options).render_markdown())


if __name__ == "__main__":
    main()
