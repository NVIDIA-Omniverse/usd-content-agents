# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record and preflight the model endpoints a pipeline run uses.

Hosted model endpoints can queue a request silently for minutes before any
byte is returned, and sometimes never serve it. A run that discovers this at
its first provider call has already spent renders and time, and fails with a
generic timeout. Pipelines use this module to record which endpoint serves
each step, and to send one small request to each distinct endpoint before the
first step so a saturated endpoint fails the run up front with a typed,
explained status.

Each agent supplies its active model plan: the models its steps will actually
call, each tagged with the interface that builds it (``vlm`` or ``llm``) and
with the config exactly as that agent's provisioning resolves it. Agents that
provision through ``ModelProvisioningTask`` resolve with
:func:`resolve_provisioned_model_config`, which applies the same NIM base-URL
environment override; agents that call the model factories directly pass
their config unchanged. The preflight never substitutes another endpoint; it
only reports whether the configured one answered within its configured
deadline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, NamedTuple
from urllib.parse import urlparse

from world_understanding.functions.models.request_timeout import (
    DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS,
)
from world_understanding.utils.credentials import (
    apply_llm_nim_env_override,
    apply_vlm_nim_env_override,
)

MODEL_ENDPOINT_PREFLIGHT_ENV = "WU_MODEL_ENDPOINT_PREFLIGHT"
PREFLIGHT_PROMPT = "Reply with the single word OK."
PREFLIGHT_MAX_TOKENS = 64
CONFIGURED_ENDPOINT_PLACEHOLDER = "configured_endpoint"
_PROVIDER_DEFAULT_HOSTS = {
    "nim": "integrate.api.nvidia.com",
    "openai": "api.openai.com",
}
UNPROBED_BACKENDS = frozenset({"mock", "echo"})
# Config fields that locate an endpoint, in the order the host is taken from.
# Backend-specific nested sections (for example a gateway mapping) are searched
# for the same keys plus ``proxy_base_url``.
_ROUTE_URL_KEYS = ("base_url", "endpoint", "azure_endpoint")
_NESTED_ROUTE_URL_KEYS = ("azure_endpoint", "proxy_base_url", "base_url", "endpoint")
# Fields that never change which endpoint is called: request settings and
# credentials. Every other field joins the private route identity.
_NON_ROUTE_KEYS = frozenset(
    {
        "timeout",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "max_retries",
        "generation_kwargs",
        "enabled",
        "reference_images",
        "api_key",
        "api_key_env",
        "cred_dict",
    }
)

logger = logging.getLogger(__name__)


class ActiveModel(NamedTuple):
    """One model a pipeline step may call at runtime.

    ``probe_at_startup`` is false for a model the step calls only under a
    data-dependent condition (for example an adjudication model used only when
    a topology conflict appears). It is recorded, but the startup preflight
    does not probe it, so an endpoint the run may never call cannot abort it.

    ``required`` is false for a model the step can do without, such as the
    prediction parsing LLM that falls back to the VLM. It is probed, but a
    failure is recorded as ``optional_unavailable`` instead of aborting.
    """

    step: str
    section: str
    interface: Literal["vlm", "llm"]
    config: Mapping[str, Any]
    probe_at_startup: bool = True
    required: bool = True


class ModelEndpointPreflightError(RuntimeError):
    """A configured model endpoint did not answer the preflight request."""

    def __init__(self, message: str, status: dict[str, Any]) -> None:
        super().__init__(message)
        self.status = status


def model_endpoint_preflight_enabled(default: bool) -> bool:
    """Honor ``WU_MODEL_ENDPOINT_PREFLIGHT=0``/``1`` over the caller default."""

    raw = os.environ.get(MODEL_ENDPOINT_PREFLIGHT_ENV)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def configured_model_section(section: object) -> bool:
    """Whether a model config section names a backend (is in use)."""

    return isinstance(section, Mapping) and bool(
        section.get("backend") or section.get("provider")
    )


def endpoint_host(backend: str, base_url: object) -> str:
    """The endpoint's hostname and port only, never URL userinfo, path or query."""

    if not (isinstance(base_url, str) and base_url.strip()):
        return _PROVIDER_DEFAULT_HOSTS.get(backend, "provider_default")
    try:
        parsed = urlparse(base_url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return CONFIGURED_ENDPOINT_PLACEHOLDER
    if not hostname:
        return CONFIGURED_ENDPOINT_PLACEHOLDER
    return f"{hostname}:{port}" if port is not None else hostname


def route_url(config: Mapping[str, Any], backend: str | None = None) -> object:
    """The endpoint URL a provisioned client calls.

    A nested section named after the backend (for example a gateway section
    whose name prefixes the backend name) configures that backend's effective
    route, so its URL wins over generic top-level fields. Otherwise top-level
    route fields are used, then any other nested section.
    """

    def nested_url(section: object) -> object:
        if isinstance(section, Mapping):
            for key in _NESTED_ROUTE_URL_KEYS:
                value = section.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    if backend:
        for key, section in config.items():
            if isinstance(key, str) and key and backend.startswith(key):
                value = nested_url(section)
                if value is not None:
                    return value
    for key in _ROUTE_URL_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for section in config.values():
        value = nested_url(section)
        if value is not None:
            return value
    return None


def _route_identity(config: Mapping[str, Any]) -> str:
    """A private digest of every field that selects the endpoint (never stored)."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(k): strip(v) for k, v in value.items() if k not in _NON_ROUTE_KEYS
            }
        return value

    payload = json.dumps(strip(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_provisioned_model_config(
    interface: Literal["vlm", "llm"], config: Mapping[str, Any]
) -> dict[str, Any]:
    """A model config as ``ModelProvisioningTask`` resolves it (NIM URL override)."""

    if interface == "vlm":
        return apply_vlm_nim_env_override(dict(config))
    return apply_llm_nim_env_override(dict(config))


def describe_model_endpoints(
    active_models: Iterable[ActiveModel],
) -> list[dict[str, Any]]:
    """One record per active model, from the config the agent resolved.

    Records carry private ``_config`` and ``_route`` fields for the preflight;
    use :func:`public_endpoint_records` before persisting or logging them.
    """

    records: list[dict[str, Any]] = []
    for model in active_models:
        resolved = dict(model.config)
        backend = str(resolved.get("backend") or resolved.get("provider"))
        records.append(
            {
                "step": model.step,
                "section": model.section,
                "interface": model.interface,
                "backend": backend,
                "model": resolved.get("model"),
                "endpoint_host": endpoint_host(backend, route_url(resolved, backend)),
                "preflight": "startup" if model.probe_at_startup else "deferred",
                "required": model.required,
                "timeout_seconds": resolved.get(
                    "timeout", DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
                ),
                "_config": resolved,
                "_route": _route_identity(resolved),
            }
        )
    return records


def public_endpoint_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Endpoint records without provisioning config or route digests."""

    return [
        {k: v for k, v in record.items() if not k.startswith("_")} for record in records
    ]


def run_model_endpoint_preflight(
    records: Iterable[Mapping[str, Any]],
    *,
    create_vlm: Callable[[dict[str, Any]], Any],
    create_llm: Callable[[dict[str, Any]], Any],
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Send one small request to each distinct endpoint; raise on the first failure.

    Endpoints are distinct by interface, backend, model and full configured
    route. The request uses the section's configured deadline, so the measured
    latency reflects the endpoint as the run will see it. Status fields and the
    error message carry only value-safe fields: never the base URL, credentials
    or the provider's error text.
    """

    def identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            record.get("interface"),
            record["backend"],
            record.get("model"),
            record.get("_route"),
        )

    startup = [r for r in records if r.get("preflight") != "deferred"]
    # One probe per endpoint; it is required if any step needs it.
    required_ids = {identity(r) for r in startup if r.get("required", True)}
    users: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in startup:
        users.setdefault(identity(r), []).append(
            {"step": r.get("step"), "section": r.get("section")}
        )
    results: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in startup:
        backend = record["backend"]
        key = identity(record)
        if key in seen:
            continue
        seen.add(key)
        summary = {
            "interface": record.get("interface"),
            "backend": backend,
            "model": record.get("model"),
            "endpoint_host": record.get("endpoint_host"),
            "timeout_seconds": record.get("timeout_seconds"),
            "first_step": record.get("step"),
        }
        if backend in UNPROBED_BACKENDS:
            results.append({**summary, "outcome": "not_probed"})
            continue
        config = dict(record["_config"])
        started = clock()
        try:
            if record.get("interface") == "llm":
                create_llm(config).invoke(PREFLIGHT_PROMPT)
            else:
                create_vlm(config).generate(
                    prompt=PREFLIGHT_PROMPT, max_tokens=PREFLIGHT_MAX_TOKENS
                )
        except Exception as error:  # every provider outcome becomes a status
            elapsed = round(clock() - started, 3)
            if key not in required_ids:
                # The step falls back without this model (for example the
                # parsing LLM falls back to the VLM), so it must not abort.
                logger.warning(
                    "optional model endpoint %s %s at %s is unavailable (%s); "
                    "%s will run without it",
                    backend,
                    record.get("model"),
                    record.get("endpoint_host"),
                    type(error).__name__,
                    ", ".join(f"{u['step']}.{u['section']}" for u in users[key]),
                )
                results.append(
                    {
                        **summary,
                        "outcome": "optional_unavailable",
                        # Steps that must run without this model, so their
                        # fallback (for example the VLM) is used instead.
                        "unavailable_for": users[key],
                        "error_type": type(error).__name__,
                        "elapsed_seconds": elapsed,
                    }
                )
                continue
            status = {
                **summary,
                "outcome": "failed",
                "error_type": type(error).__name__,
                "elapsed_seconds": elapsed,
                "failed_step": "model_endpoint_preflight",
                "success": False,
                "terminal": True,
                "recovery_action": {
                    "kind": "reconfigure",
                    "detail": (
                        "The configured model endpoint did not answer a one-line "
                        "request within its deadline. Hosted endpoints can queue "
                        "requests for minutes; choose a provisioned endpoint "
                        "(the step's model base_url), raise timeout, or retry "
                        "later. No pipeline step ran."
                    ),
                },
            }
            raise ModelEndpointPreflightError(
                f"model endpoint preflight failed for {backend} "
                f"{record.get('model')} at {record.get('endpoint_host')} after "
                f"{elapsed}s ({type(error).__name__})",
                status,
            ) from None
        results.append(
            {
                **summary,
                "outcome": "answered",
                "elapsed_seconds": round(clock() - started, 3),
            }
        )
    return results
