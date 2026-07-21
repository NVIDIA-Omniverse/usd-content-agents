# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for agent-service Compose GPU pinning."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEXTURE_STEP1X_PACKAGE = "apps/texture_agent_service/docker-compose.step1x-package.yml"
_HAS_TEXTURE_STEP1X_PACKAGE = (REPO_ROOT / _TEXTURE_STEP1X_PACKAGE).exists()
TEXTURE_PLANNING_CAP_COMPOSE_PATHS = [
    "apps/texture_agent_service/docker-compose.yml",
    "apps/texture_agent_service/docker-compose.step1x.yml",
]
if _HAS_TEXTURE_STEP1X_PACKAGE:
    TEXTURE_PLANNING_CAP_COMPOSE_PATHS.append(_TEXTURE_STEP1X_PACKAGE)
_INTERNAL_TEXTURE_PACKAGE = (
    "apps/texture_agent_service/internal/docker-compose.step1x-package.yml"
)
if (REPO_ROOT / _INTERNAL_TEXTURE_PACKAGE).exists():
    TEXTURE_PLANNING_CAP_COMPOSE_PATHS.append(_INTERNAL_TEXTURE_PACKAGE)
TEXTURE_RENDERED_PLANNING_CAP_COMPOSE_PATHS = [
    ("apps/texture_agent_service/docker-compose.yml",),
    (
        "apps/texture_agent_service/docker-compose.yml",
        "apps/texture_agent_service/docker-compose.step1x.yml",
    ),
]
if _HAS_TEXTURE_STEP1X_PACKAGE:
    TEXTURE_RENDERED_PLANNING_CAP_COMPOSE_PATHS.append((_TEXTURE_STEP1X_PACKAGE,))
if (REPO_ROOT / _INTERNAL_TEXTURE_PACKAGE).exists():
    TEXTURE_RENDERED_PLANNING_CAP_COMPOSE_PATHS.append((_INTERNAL_TEXTURE_PACKAGE,))
OVRTX_DAEMON_LIMIT_COMPOSE_PATHS = [
    "apps/ovrtx_rendering_api/docker-compose.yml",
    "apps/material_agent_service/docker-compose.yml",
    "apps/physics_agent_service/docker-compose.yml",
    "apps/texture_agent_service/docker-compose.step1x.yml",
]
if _HAS_TEXTURE_STEP1X_PACKAGE:
    OVRTX_DAEMON_LIMIT_COMPOSE_PATHS.append(_TEXTURE_STEP1X_PACKAGE)
if (REPO_ROOT / _INTERNAL_TEXTURE_PACKAGE).exists():
    OVRTX_DAEMON_LIMIT_COMPOSE_PATHS.append(_INTERNAL_TEXTURE_PACKAGE)

_requires_step1x_package = pytest.mark.skipif(
    not _HAS_TEXTURE_STEP1X_PACKAGE,
    reason="managed Step1X runtime package is not shipped in this artifact",
)


class ComposeLoader(yaml.SafeLoader):
    """YAML loader that treats Compose merge tags as their underlying value."""


def _construct_override(loader: ComposeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ComposeLoader.add_constructor("!override", _construct_override)


def _load_compose(path: str) -> dict[str, Any]:
    with (REPO_ROOT / path).open(encoding="utf-8") as f:
        return yaml.load(f, Loader=ComposeLoader)


def _load_merged_compose(
    *paths: str,
    profile: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if shutil.which("docker") is None:
        pytest.skip("docker is required to validate the merged Compose config")

    cmd = ["docker", "compose"]
    for path in paths:
        cmd.extend(["-f", path])
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["config", "--no-interpolate", "--format", "json"])

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return json.loads(result.stdout)


def _load_rendered_compose(*paths: str) -> dict[str, Any]:
    """Render Compose defaults without inheriting a developer's repo .env."""
    if shutil.which("docker") is None:
        pytest.skip("docker is required to validate the rendered Compose config")

    cmd = ["docker", "compose", "--env-file", "/dev/null"]
    for path in paths:
        cmd.extend(["-f", path])
    cmd.extend(["config", "--format", "json"])
    render_env = dict(os.environ)
    for variable in (
        "TA_TEXTURE_PLAN_DEFAULT_CAP",
        "TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP",
        "TA_TEXTURE_PLAN_HARD_CAP",
        "TA_MAX_TEXTURE_UNITS",
    ):
        render_env.pop(variable, None)
    render_env["TEXTURE_STEP1X_HOST_RUNTIME"] = str(REPO_ROOT)

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=render_env,
    )
    return json.loads(result.stdout)


def _reservation_devices(service: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        service.get("deploy", {})
        .get("resources", {})
        .get("reservations", {})
        .get("devices", [])
    )


def _assert_texture_planning_cap_environment(environment: list[str]) -> None:
    """Verify one Compose profile exposes the immutable plan-v1 limits."""
    assert (
        "TA_TEXTURE_PLAN_DEFAULT_CAP=${TA_TEXTURE_PLAN_DEFAULT_CAP:-32}" in environment
    )
    assert (
        "TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP="
        "${TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP:-16}"
    ) in environment
    assert "TA_TEXTURE_PLAN_HARD_CAP=${TA_TEXTURE_PLAN_HARD_CAP:-64}" in environment
    assert "TA_MAX_TEXTURE_UNITS=${TA_MAX_TEXTURE_UNITS:-64}" in environment


@pytest.mark.parametrize("compose_path", OVRTX_DAEMON_LIMIT_COMPOSE_PATHS)
def test_ovrtx_compose_profiles_bound_daemon_lifetime(compose_path: str) -> None:
    compose = _load_compose(compose_path)
    environment = compose["services"]["ovrtx-rendering-api"]["environment"]

    assert "OVRTX_DAEMON_MAX_RENDERS=${OVRTX_DAEMON_MAX_RENDERS:-64}" in environment
    assert (
        "OVRTX_DAEMON_MAX_RSS_BYTES=${OVRTX_DAEMON_MAX_RSS_BYTES:-25769803776}"
        in environment
    )


@pytest.mark.parametrize(
    "compose_path",
    TEXTURE_PLANNING_CAP_COMPOSE_PATHS,
)
def test_texture_compose_profiles_publish_bounded_planning_caps(
    compose_path: str,
) -> None:
    compose = _load_compose(compose_path)

    environment = compose["services"]["texture-agent-service"]["environment"]

    _assert_texture_planning_cap_environment(environment)


@pytest.mark.parametrize("compose_paths", TEXTURE_RENDERED_PLANNING_CAP_COMPOSE_PATHS)
def test_rendered_texture_compose_profiles_use_contract_planning_defaults(
    compose_paths: tuple[str, ...],
) -> None:
    compose = _load_rendered_compose(*compose_paths)

    environment = compose["services"]["texture-agent-service"]["environment"]

    assert environment["TA_TEXTURE_PLAN_DEFAULT_CAP"] == "32"
    assert environment["TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP"] == "16"
    assert environment["TA_TEXTURE_PLAN_HARD_CAP"] == "64"
    assert environment["TA_MAX_TEXTURE_UNITS"] == "64"


def test_physics_multi_gpu_overlay_routes_to_local_vlm_nim() -> None:
    compose = _load_compose("apps/physics_agent_service/docker-compose.multi-gpu.yml")

    environment = compose["services"]["physics-agent-service"]["environment"]

    assert "PA_VLM_NIM_BASE_URL=http://vlm-nim:8000/v1" in environment
    assert "PA_VLM_MODEL=nvidia/cosmos-reason2-8b" in environment
    assert "PA_LLM_NIM_BASE_URL=http://vlm-nim:8000/v1" in environment
    assert "PA_NIM_API_KEY=not-used" in environment


def test_physics_multi_gpu_overlay_pins_sidecars_to_separate_gpus() -> None:
    compose = _load_compose("apps/physics_agent_service/docker-compose.multi-gpu.yml")

    ovrtx_devices = _reservation_devices(compose["services"]["ovrtx-rendering-api"])
    vlm_devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert ovrtx_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["0"],
            "capabilities": ["gpu"],
        }
    ]
    assert vlm_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_material_vlm_nim_base_file_does_not_pregrant_gpu() -> None:
    """Keep GPU selection out of the base service to avoid Compose list append."""
    compose = _load_compose("apps/material_agent_service/docker-compose.yml")

    assert _reservation_devices(compose["services"]["vlm-nim"]) == []


def test_material_multi_gpu_overlay_pins_vlm_nim_to_gpu_1() -> None:
    compose = _load_compose("apps/material_agent_service/docker-compose.multi-gpu.yml")

    devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_material_multi_gpu_merged_config_replaces_base_gpu_reservations() -> None:
    compose = _load_merged_compose(
        "apps/material_agent_service/docker-compose.yml",
        "apps/material_agent_service/docker-compose.multi-gpu.yml",
        profile="vlm",
    )

    ovrtx_devices = _reservation_devices(compose["services"]["ovrtx-rendering-api"])
    vlm_devices = _reservation_devices(compose["services"]["vlm-nim"])

    assert ovrtx_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["0"],
            "capabilities": ["gpu"],
        }
    ]
    assert vlm_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


def test_texture_multi_gpu_overlay_uses_local_image_gen_placeholder_key() -> None:
    compose = _load_compose("apps/texture_agent_service/docker-compose.multi-gpu.yml")

    environment = compose["services"]["texture-agent-service"]["environment"]

    assert "TA_IMAGE_GEN_BACKEND=openai" in environment
    assert "TA_IMAGE_GEN_BASE_URL=http://image-gen-nim:8000/v1" in environment
    assert "TA_IMAGE_GEN_API_KEY=not-used" in environment


def test_texture_step1x_overlay_routes_to_step1x_and_ovrtx_sidecars() -> None:
    compose = _load_compose("apps/texture_agent_service/docker-compose.step1x.yml")

    service = compose["services"]["texture-agent-service"]
    environment = service["environment"]

    assert "TA_TEXTURE_BACKEND=service" in environment
    assert "TA_TEXTURE_ENDPOINT=http://texture-gen-step1x:8000" in environment
    assert "TA_BACKEND_ENGINE=step1x" in environment
    assert "TA_SIMPLE_TEXTURE_ENDPOINT=http://texture-gen-simple:8000" in environment
    assert "TA_SIMPLE_BACKEND_ENGINE=simple_image_gen" in environment
    assert "TA_SIMPLE_UV_SCOPE=${TA_SIMPLE_UV_SCOPE:-stage}" in environment
    assert "TA_SIMPLE_TEXTURE_WORKERS=${TEXTURE_GEN_MAX_WORKERS:-4}" in environment
    assert "TA_UV_POLICY=${TA_UV_POLICY:-generate_missing}" in environment
    assert "TA_UV_OVERWRITE_EXISTING=${TA_UV_OVERWRITE_EXISTING:-false}" in environment
    assert "TA_RENDER_ENABLED=${TA_RENDER_ENABLED:-true}" in environment
    assert "RENDER_ENDPOINT=http://ovrtx-rendering-api:8000" in environment
    assert service["depends_on"]["texture-gen-simple"]["condition"] == (
        "service_healthy"
    )
    assert service["depends_on"]["texture-gen-step1x"]["condition"] == (
        "service_healthy"
    )
    assert service["depends_on"]["ovrtx-rendering-api"]["condition"] == (
        "service_healthy"
    )
    step1x = compose["services"]["texture-gen-step1x"]
    assert (
        "TEXTURE_STEP1X_REQUIRED_EXECUTABLES=${TEXTURE_STEP1X_REQUIRED_EXECUTABLES:-uv}"
    ) in step1x["environment"]
    assert (
        "TEXTURE_STEP1X_REQUIRE_UPSCALER=${TEXTURE_STEP1X_REQUIRE_UPSCALER:-false}"
    ) in step1x["environment"]
    assert (
        "TEXTURE_STEP1X_EDIT_SCRIPT=${TEXTURE_STEP1X_EDIT_SCRIPT:-/opt/texture-editing/edit_texture.py}"
        in step1x["environment"]
    )
    assert (
        "TEXTURE_UPSCALER_BACKEND=${TEXTURE_UPSCALER_BACKEND:-swin2sr}"
        in step1x["environment"]
    )
    assert (
        "TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS=${TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS:-true}"
        in step1x["environment"]
    )
    assert any(
        value.startswith("LD_LIBRARY_PATH=${TEXTURE_STEP1X_LD_LIBRARY_PATH:-")
        for value in step1x["environment"]
    )
    assert (
        "NVIDIA_VISIBLE_DEVICES=${TEXTURE_STEP1X_GPU_DEVICE:-0}"
        in step1x["environment"]
    )
    assert step1x["healthcheck"]["test"] == ["CMD", "python3", "/healthcheck.py"]
    assert (
        step1x["healthcheck"]["timeout"]
        == "${TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT:-180s}"
    )
    assert _reservation_devices(step1x) == [
        {
            "driver": "nvidia",
            "device_ids": ["${TEXTURE_STEP1X_GPU_DEVICE:-0}"],
            "capabilities": ["gpu"],
        }
    ]
    simple = compose["services"]["texture-gen-simple"]
    assert simple["image"] == "texture-agent-service:local"
    assert "apps.texture_gen_simple_service.app:app" in simple["command"]
    assert (
        "TEXTURE_OUTPUT_DIR=/var/texture-agent/sessions/texture_gen_simple_outputs"
        in simple["environment"]
    )
    assert {
        "path": "${TEXTURE_GEN_SIMPLE_ENV_FILE:-/dev/null}",
        "required": False,
    } in simple["env_file"]
    assert (
        "TEXTURE_GEN_BACKEND=${TEXTURE_GEN_BACKEND:-nim}" not in simple["environment"]
    )
    assert "TEXTURE_GEN_MODEL=${TEXTURE_GEN_MODEL:-}" not in simple["environment"]
    assert "TEXTURE_GEN_BASE_URL=${TEXTURE_GEN_BASE_URL:-}" not in simple["environment"]
    assert "TEXTURE_GEN_API_KEY=${TEXTURE_GEN_API_KEY:-}" not in simple["environment"]
    assert (
        "TEXTURE_GEN_MAX_WORKERS=${TEXTURE_GEN_MAX_WORKERS:-4}" in simple["environment"]
    )
    assert simple["healthcheck"]["test"] == [
        "CMD",
        "curl",
        "-f",
        "http://localhost:8000/livez",
    ]
    ovrtx = compose["services"]["ovrtx-rendering-api"]
    assert "NVIDIA_VISIBLE_DEVICES=${OVRTX_GPU_DEVICE:-1}" in ovrtx["environment"]
    assert _reservation_devices(ovrtx) == [
        {
            "driver": "nvidia",
            "device_ids": ["${OVRTX_GPU_DEVICE:-1}"],
            "capabilities": ["gpu"],
        }
    ]


def test_texture_step1x_overlay_shares_session_volume_with_step1x() -> None:
    compose = _load_compose("apps/texture_agent_service/docker-compose.step1x.yml")

    volumes = compose["services"]["texture-gen-step1x"]["volumes"]

    assert "session-storage:/var/texture-agent/sessions" in volumes


def test_texture_step1x_standalone_compose_has_runtime_health_preflight() -> None:
    compose = _load_compose("apps/texture_gen_step1x_service/docker-compose.yml")

    step1x = compose["services"]["texture-gen-step1x"]
    environment = step1x["environment"]

    assert (
        environment["TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS"]
        == "${TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS:-true}"
    )
    assert environment["LD_LIBRARY_PATH"].startswith(
        "${TEXTURE_STEP1X_LD_LIBRARY_PATH:-"
    )
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "${TEXTURE_STEP1X_GPU_DEVICE:-0}"
    assert step1x["healthcheck"]["test"] == ["CMD", "python3", "/healthcheck.py"]
    assert (
        step1x["healthcheck"]["timeout"]
        == "${TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT:-180s}"
    )
    assert _reservation_devices(step1x) == [
        {
            "driver": "nvidia",
            "device_ids": ["${TEXTURE_STEP1X_GPU_DEVICE:-0}"],
            "capabilities": ["gpu"],
        }
    ]


def test_texture_step1x_multi_gpu_overlay_pins_step1x_and_ovrtx() -> None:
    compose = _load_compose(
        "apps/texture_agent_service/docker-compose.step1x.multi-gpu.yml"
    )

    step1x_devices = _reservation_devices(compose["services"]["texture-gen-step1x"])
    ovrtx_devices = _reservation_devices(compose["services"]["ovrtx-rendering-api"])

    assert step1x_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["0"],
            "capabilities": ["gpu"],
        }
    ]
    assert ovrtx_devices == [
        {
            "driver": "nvidia",
            "device_ids": ["1"],
            "capabilities": ["gpu"],
        }
    ]


@_requires_step1x_package
def test_texture_step1x_package_routes_and_pins_gpu_sidecars() -> None:
    compose = _load_compose(_TEXTURE_STEP1X_PACKAGE)

    service = compose["services"]["texture-agent-service"]
    environment = service["environment"]
    step1x = compose["services"]["texture-gen-step1x"]
    runtime_setup = compose["services"]["texture-step1x-runtime-setup"]
    ovrtx = compose["services"]["ovrtx-rendering-api"]
    simple = compose["services"]["texture-gen-simple"]

    assert "TA_TEXTURE_BACKEND=service" in environment
    assert "TA_TEXTURE_ENDPOINT=http://texture-gen-step1x:8000" in environment
    assert "TA_BACKEND_ENGINE=step1x" in environment
    assert "TA_SIMPLE_TEXTURE_ENDPOINT=http://texture-gen-simple:8000" in environment
    assert "TA_SIMPLE_BACKEND_ENGINE=simple_image_gen" in environment
    assert "TA_SIMPLE_UV_SCOPE=${TA_SIMPLE_UV_SCOPE:-stage}" in environment
    assert "TA_SIMPLE_TEXTURE_WORKERS=${TEXTURE_GEN_MAX_WORKERS:-4}" in environment
    assert "TA_UV_POLICY=${TA_UV_POLICY:-generate_missing}" in environment
    assert "TA_UV_SCOPE=${TA_UV_SCOPE:-stage}" in environment
    assert (
        "TA_UV_REBAKE_SOURCE_ALBEDO=${TA_UV_REBAKE_SOURCE_ALBEDO:-true}" in environment
    )
    assert "TA_UV_REBAKE_SIZE=${TA_UV_REBAKE_SIZE:-2048}" in environment
    assert "TA_UV_OVERWRITE_EXISTING=${TA_UV_OVERWRITE_EXISTING:-false}" in environment
    assert "TA_RENDER_ENABLED=${TA_RENDER_ENABLED:-true}" in environment
    assert "RENDER_ENDPOINT=http://ovrtx-rendering-api:8000" in environment
    assert service["depends_on"]["texture-gen-simple"]["condition"] == (
        "service_healthy"
    )
    assert service["depends_on"]["texture-gen-step1x"]["condition"] == (
        "service_healthy"
    )
    assert service["depends_on"]["ovrtx-rendering-api"]["condition"] == (
        "service_healthy"
    )
    assert (
        "TEXTURE_STEP1X_REQUIRED_EXECUTABLES=${TEXTURE_STEP1X_REQUIRED_EXECUTABLES-uv}"
    ) in step1x["environment"]
    assert ("TEXTURE_STEP1X_SKIP_MA=${TEXTURE_STEP1X_SKIP_MA:-true}") in step1x[
        "environment"
    ]
    assert (
        "TEXTURE_STEP1X_REQUIRE_UPSCALER=${TEXTURE_STEP1X_REQUIRE_UPSCALER:-false}"
    ) in step1x["environment"]
    assert ("TEXTURE_STEP1X_SKIP_MA=${TEXTURE_STEP1X_SKIP_MA:-true}") in runtime_setup[
        "environment"
    ]
    assert (
        "TEXTURE_STEP1X_REQUIRE_UPSCALER=${TEXTURE_STEP1X_REQUIRE_UPSCALER:-false}"
    ) in runtime_setup["environment"]
    assert (
        "TEXTURE_STEP1X_ACCEPT_RESTRICTED_RUNTIME_LICENSES="
        "${TEXTURE_STEP1X_ACCEPT_RESTRICTED_RUNTIME_LICENSES:-false}"
    ) in runtime_setup["environment"]
    assert (
        "TEXTURE_UPSCALER_BACKEND=${TEXTURE_UPSCALER_BACKEND:-swin2sr}"
        in step1x["environment"]
    )
    assert (
        "TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS=${TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS:-true}"
        in step1x["environment"]
    )
    assert any(
        value.startswith("LD_LIBRARY_PATH=${TEXTURE_STEP1X_LD_LIBRARY_PATH:-")
        for value in step1x["environment"]
    )
    assert step1x["healthcheck"]["test"] == [
        "CMD",
        "python3",
        "/healthcheck.py",
        "--liveness",
    ]
    assert (
        step1x["healthcheck"]["timeout"]
        == "${TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT:-180s}"
    )
    assert "session-storage:/var/texture-agent/sessions" in step1x["volumes"]
    assert simple["image"] == "texture-agent-service:step1x-package"
    assert "apps.texture_gen_simple_service.app:app" in simple["command"]
    assert (
        "TEXTURE_OUTPUT_DIR=/var/texture-agent/sessions/texture_gen_simple_outputs"
        in simple["environment"]
    )
    assert {
        "path": "${TEXTURE_GEN_SIMPLE_ENV_FILE:-/dev/null}",
        "required": False,
    } in simple["env_file"]
    assert (
        "TEXTURE_GEN_BACKEND=${TEXTURE_GEN_BACKEND:-nim}" not in simple["environment"]
    )
    assert "TEXTURE_GEN_MODEL=${TEXTURE_GEN_MODEL:-}" not in simple["environment"]
    assert "TEXTURE_GEN_BASE_URL=${TEXTURE_GEN_BASE_URL:-}" not in simple["environment"]
    assert "TEXTURE_GEN_API_KEY=${TEXTURE_GEN_API_KEY:-}" not in simple["environment"]
    assert (
        "TEXTURE_GEN_MAX_WORKERS=${TEXTURE_GEN_MAX_WORKERS:-4}" in simple["environment"]
    )
    assert "session-storage:/var/texture-agent/sessions" in simple["volumes"]
    assert simple["healthcheck"]["test"] == [
        "CMD",
        "curl",
        "-f",
        "http://localhost:8000/livez",
    ]
    assert _reservation_devices(step1x) == [
        {
            "driver": "nvidia",
            "device_ids": ["${TEXTURE_STEP1X_GPU_DEVICE:-0}"],
            "capabilities": ["gpu"],
        }
    ]
    assert _reservation_devices(ovrtx) == [
        {
            "driver": "nvidia",
            "device_ids": ["${OVRTX_GPU_DEVICE:-1}"],
            "capabilities": ["gpu"],
        }
    ]


@_requires_step1x_package
def test_texture_step1x_package_merged_config_parses_with_required_env() -> None:
    compose = _load_merged_compose(
        _TEXTURE_STEP1X_PACKAGE,
        env={"TEXTURE_STEP1X_HOST_RUNTIME": str(REPO_ROOT)},
    )

    assert "texture-agent-service" in compose["services"]
    assert "texture-gen-simple" in compose["services"]
    assert "texture-gen-step1x" in compose["services"]
    assert "ovrtx-rendering-api" in compose["services"]


@_requires_step1x_package
def test_texture_step1x_package_fake_runner_path_is_copied_into_image() -> None:
    compose_text = (REPO_ROOT / _TEXTURE_STEP1X_PACKAGE).read_text(encoding="utf-8")
    smoke_runner = (
        REPO_ROOT / "apps/texture_gen_step1x_service/smoke/fake_step1x_runner.py"
    )

    assert str(smoke_runner.relative_to(REPO_ROOT)) in compose_text
    assert smoke_runner.exists()


def test_texture_step1x_healthcheck_script_is_packaged_and_used() -> None:
    healthcheck = REPO_ROOT / "apps/texture_gen_step1x_service/healthcheck.py"
    public_dockerfile = (
        REPO_ROOT / "apps/texture_gen_step1x_service/Dockerfile"
    ).read_text(encoding="utf-8")
    internal_dockerfile_path = (
        REPO_ROOT / "apps/texture_gen_step1x_service/internal/Dockerfile.runtime"
    )

    assert healthcheck.exists()
    dockerfiles = [public_dockerfile]
    if internal_dockerfile_path.exists():
        dockerfiles.append(internal_dockerfile_path.read_text(encoding="utf-8"))

    for dockerfile in dockerfiles:
        assert (
            "apps/texture_gen_step1x_service/healthcheck.py /healthcheck.py"
            in dockerfile
        )

    assert "--chown=root:root --chmod=0555" in public_dockerfile
    assert 'CMD ["/opt/venv/bin/python", "/healthcheck.py", "--liveness"]' in (
        public_dockerfile
    )
    if internal_dockerfile_path.exists():
        internal_dockerfile = internal_dockerfile_path.read_text(encoding="utf-8")
        assert "chmod 0755 /healthcheck.py" in internal_dockerfile
        assert "CMD python3 /healthcheck.py" in internal_dockerfile

    compose_paths = [
        "apps/texture_agent_service/docker-compose.step1x.yml",
        "apps/texture_gen_step1x_service/docker-compose.yml",
    ]
    if _HAS_TEXTURE_STEP1X_PACKAGE:
        compose_paths.append(_TEXTURE_STEP1X_PACKAGE)
    optional_compose_paths = [
        "apps/texture_agent_service/internal/docker-compose.step1x-package.yml",
        "apps/texture_gen_step1x_service/docker-compose.internal.yml",
    ]
    compose_paths.extend(
        compose_path
        for compose_path in optional_compose_paths
        if (REPO_ROOT / compose_path).exists()
    )
    for compose_path in compose_paths:
        step1x = _load_compose(compose_path)["services"]["texture-gen-step1x"]
        expected = ["CMD", "python3", "/healthcheck.py"]
        if compose_path == _TEXTURE_STEP1X_PACKAGE:
            expected.append("--liveness")
        assert step1x["healthcheck"]["test"] == expected
