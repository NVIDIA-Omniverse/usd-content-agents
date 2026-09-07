# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote render backend helpers (client-side).

Two operations, both bypass command dispatch (they manage local config / probe a remote
service, not the scene):

  * `serve_cmd`  — print the exact commands to bring the managed OVRTX adapter up on a
    remote GPU box (`apps/ovrtx_rendering_api`).
  * `configure`  — write `renderer = "remote"` + `remote_url` into a config.toml, then
    probe the service's `/health` so a misconfigured URL fails loudly, here, not later
    at render time.

The service multiplexes many clients (it's stateless — each /render carries its own USD)
but renders serially (one GPU daemon behind a lock), so `configure` reports that too.
"""

from __future__ import annotations

from pathlib import Path

from usd_core.config import (
    CONFIG_NAME,
    Config,
    find_project_dir,
    _read_toml,
    state_dir_for,
)


def _global_config_path() -> Path:
    return Path.home() / ".config" / "usd-cli" / CONFIG_NAME


def _project_config_path(config: Config) -> Path | None:
    # state_dir_for: a pre-rename project keeps writing .3dsc/config.toml — the
    # same file the daemon reads — instead of forking a parallel .usd-cli/ tree
    base = config.project_dir or find_project_dir()
    return (state_dir_for(base) / CONFIG_NAME) if base else None


def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def _dump_toml(data: dict) -> str:
    """Serialize a 1-level-of-tables config dict (scalars + named tables). Sufficient for
    usd-cli config.toml — it has no arrays or nested tables. Existing keys are preserved;
    comments are not (tomllib can't round-trip them)."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines: list[str] = [f"{k} = {_toml_value(v)}" for k, v in scalars.items()]
    for name, tbl in tables.items():
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        lines.extend(f"{k} = {_toml_value(v)}" for k, v in tbl.items())
    return "\n".join(lines) + "\n"


def serve_cmd(*, bare: bool = False, port: int = 8000) -> str:
    """Instructions to run the managed OVRTX adapter on a remote GPU host."""
    docker = (
        "# On the remote Linux + NVIDIA RTX box (needs the NVIDIA container runtime):\n"
        "git clone <content-agents-repo> content-agents && cd content-agents\n"
        "export OVRTX_API_KEY='<strong-random-secret>'\n"
        "export CUDA_BASE_IMAGE='nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:<approved-digest>'\n"
        "USD_CLI_ROOT=apps/usd_cli\n"
        "SERVICE_DIR=\"$USD_CLI_ROOT/apps/ovrtx_rendering_api\"\n"
        f"PORT={port} docker compose -f \"$SERVICE_DIR/docker-compose.yml\" up --build -d\n"
        "\n"
        f"# Verify it's reachable (from here or there):\n"
        f"curl -H \"Authorization: Bearer $OVRTX_API_KEY\" https://<remote-host>:{port}/ready\n"
    )
    if not bare:
        return docker
    return (
        "# On the remote Linux + NVIDIA RTX box (bare metal, no Docker):\n"
        "git clone <content-agents-repo> content-agents && cd content-agents\n"
        "USD_CLI_ROOT=apps/usd_cli\n"
        "SERVICE_DIR=\"$USD_CLI_ROOT/apps/ovrtx_rendering_api\"\n"
        "python -m venv .venv && . .venv/bin/activate\n"
        "pip install -e \"$USD_CLI_ROOT\" && pip install -e \"$SERVICE_DIR\"\n"
        "export OVRTX_API_KEY='<strong-random-secret>'\n"
        "export DISPLAY=:0   # a virtual X display (Xvfb) must be running for Vulkan\n"
        f"cd \"$SERVICE_DIR\" && PORT={port} \\\n"
        f"  uvicorn service.main:app --host 127.0.0.1 --port {port}\n"
        "\n"
        "# First render provisions an isolated ovrtx venv from https://pypi.nvidia.com\n"
        "# (cold start can take minutes; /health reports gpu_initialized=false until ready).\n"
        f"\n# Verify through your TLS ingress:\ncurl -H \"Authorization: Bearer $OVRTX_API_KEY\" "
        f"https://<remote-host>:{port}/ready\n"
    )


def write_remote_url(config: Config, url: str, *, api_key: str | None = None,
                     use_global: bool = False) -> Path:
    """Persist renderer=remote + remote_url into the project (default) or global config,
    preserving any other keys already in that file. Returns the file written."""
    if use_global:
        path = _global_config_path()
    else:
        path = _project_config_path(config) or _global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_toml(path)
    render = dict(existing.get("render", {}))
    render["renderer"] = "remote"
    render["remote_url"] = url.rstrip("/")
    if api_key:
        render["remote_api_key"] = api_key
    existing["render"] = render

    path.write_text(_dump_toml(existing))
    if api_key:
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path


def probe(url: str, *, api_key: str | None = None, timeout: float = 10.0) -> dict:
    """GET {url}/health. Returns the parsed health dict, or {'error': ...} on failure."""
    import httpx  # lazy

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(f"{url.rstrip('/')}/health", headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — connectivity check, report don't raise
        return {"error": f"{type(exc).__name__}: {exc}"}
