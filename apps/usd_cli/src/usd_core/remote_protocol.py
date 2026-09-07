# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client ↔ backend version handshake for the managed OVRTX adapter.

`PROTOCOL_VERSION` is the single source of truth for the wire contract between this
checkout and `apps/ovrtx_rendering_api`. The service installs `usd_core` from the same
repo, so each deployed backend bakes in the version of the checkout it was built from
and reports it on the public `GET /live` (and in `/health`). Bump the number whenever
the contract changes — request/response models, endpoints, or their semantics.

Clients verify the backend's version before uploading anything and refuse to run
against a backend reporting a different (or no) version: a silent contract drift
produces wrong renders/simulations, not errors, so mismatches must fail loudly and
ask for a backend re-deploy. `render.remote_verify_version = false` overrides the
check for emergencies.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

# v2: render requests may carry `camera_defs` — tool-authored render cameras are
# stripped from the uploaded bundle (making it viewpoint-independent and cache-
# friendly) and re-authored by the service from these specs before rendering.
# v3: every render result carries the exact OVRTX render mode, sensor-update
# count, and active AOV reported by the backend that executed it.
PROTOCOL_VERSION = 3

_REDEPLOY_HINT = (
    "re-deploy the rendering service from this checkout "
    "(apps/ovrtx_rendering_api — `usd-cli remote serve-cmd` prints the commands), then "
    "retry. To override at your own risk, put `remote_verify_version = false` "
    "directly under `[render]` in config.toml, or export "
    "`USD_CLI_RENDER_REMOTE_VERIFY_VERSION=false`."
)


class ProtocolMismatchError(RuntimeError):
    """The remote backend speaks a different protocol version than this client."""


def check_remote_protocol(
    base_url: str,
    *,
    client: Any | None = None,
    timeout: float = 10.0,
    required_engine: str | None = None,
    required_features: Collection[str] = (),
    api_key: str | None = None,
) -> dict:
    """Verify the service at `base_url` speaks this client's protocol version.

    Probes `GET /live`, sending the configured bearer key when the hosting layer
    (for example NVCF) protects every invocation path. The service itself answers
    during GPU warm-up.
    Returns the backend's whole /live payload on match — newer services advertise
    `max_body_bytes` (so the client upload cap can adopt the backend's real limit)
    and `features` (e.g. "zstd") beside `protocol_version`; older v2 services
    return just the version and status. Raises `ProtocolMismatchError` on any
    mismatch — including backends that predate version reporting — and `RuntimeError`
    when the service cannot be reached at all.
    """
    import httpx  # lazy

    url = f"{base_url.rstrip('/')}/live"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        if client is not None:
            resp = client.get(url, headers=headers, timeout=timeout)
        else:
            resp = httpx.get(url, headers=headers, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — network layer; surface as one clear error
        raise RuntimeError(
            f"cannot reach the remote usd-cli backend at {base_url} to verify its "
            f"version ({type(exc).__name__}: {exc})") from exc

    backend_version: object = None
    info: dict = {}
    if resp.status_code == 200:
        try:
            payload = resp.json()
            if isinstance(payload, dict):
                info = payload
            backend_version = info.get("protocol_version")
        except ValueError:
            backend_version = None

    if backend_version is None:
        raise ProtocolMismatchError(
            f"the remote usd-cli backend at {base_url} does not report a protocol version "
            f"(it was deployed before version checking; this client requires "
            f"v{PROTOCOL_VERSION}) — {_REDEPLOY_HINT}")
    if backend_version != PROTOCOL_VERSION:
        # backend_version is untrusted JSON headed for a terminal: keep it short
        # and strip control characters before embedding.
        shown = "".join(c for c in str(backend_version)[:32]
                        if ord(c) >= 32 and c != "\x7f") or "?"
        raise ProtocolMismatchError(
            f"the remote usd-cli backend at {base_url} speaks protocol "
            f"v{shown} but this client requires v{PROTOCOL_VERSION} — "
            f"{_REDEPLOY_HINT}"
        )
    if required_engine is not None and info.get("engine") != required_engine:
        reported = str(info.get("engine") or "unknown")[:32]
        raise ProtocolMismatchError(
            f"the remote usd-cli backend at {base_url} reports engine "
            f"{reported!r}, but this caller requires {required_engine!r}; "
            "deploy an engine-identifying compatible backend and retry"
        )
    features = info.get("features")
    advertised_features = (
        {value for value in features if isinstance(value, str)}
        if isinstance(features, list)
        else set()
    )
    missing_features = sorted(set(required_features) - advertised_features)
    if missing_features:
        raise ProtocolMismatchError(
            f"the remote usd-cli backend at {base_url} does not advertise required "
            f"feature(s): {', '.join(missing_features)}; {_REDEPLOY_HINT}"
        )
    return info
