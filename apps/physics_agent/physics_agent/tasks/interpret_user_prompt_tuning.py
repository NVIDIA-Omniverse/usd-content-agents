# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Natural-language → :class:`Scenario` interpreter for ``physics-agent tune``.

Implements Part 1.1 of the physics-agent tune NL feature (closed issue #51):
when the user invokes ``physics-agent tune --user-prompt "make this object
bouncy"`` the runner calls :func:`infer_scenario_from_prompt` here to turn the
free-form prompt into a validated :class:`physics_agent.tuning.types.Scenario`
with biased parameter bounds.

Design contract:
    * Single LLM call (one retry on parser-validation failure).
    * Strict-JSON output validated through
      :func:`physics_agent.tuning.scenario.parse_scenario`.
    * Explicit user config (``scenario_override``) wins on every key conflict;
      this module only fills gaps left by the user's YAML.
    * Result is written (write-only audit, never read back) as
      ``inferred_scenario.json`` inside the caller-supplied
      ``audit_dir`` for human review.
    * Failures raise :class:`InterpreterError` (chained to the underlying
      cause) — the caller decides whether to fall back to the user-supplied
      config or surface the error.

Import-graph constraint
-----------------------
This module is intentionally lightweight: it MUST NOT import anything from
``physics_agent.tuning.optimizers`` (which transitively pulls in ``botorch``,
``torch``, ``ovphysx``). Bringing those into the import graph would double the
import cost of every CLI entry point that touches tuning code (e.g. ``--help``)
and would break the subprocess test that asserts these heavy dependencies are
not loaded by the NL interpreter.

Allowed imports from the tuning subpackage:
    * :mod:`physics_agent.tuning.types`     — data classes + constants
    * :mod:`physics_agent.tuning.scenario`  — ``parse_scenario`` + error type

LLM-related imports go through ``world_understanding`` only.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from world_understanding.functions.graphics.rendering_backend_factory import (
    RENDERING_BACKEND_NAMES,
)
from world_understanding.functions.nlp.chat import generate_chat_response

from physics_agent.tuning.scenario import ScenarioParseError, parse_scenario
from physics_agent.tuning.types import (
    DEFAULT_PARAM_BOUNDS,
    SUPPORTED_PARAM_KEYS,
    SUPPORTED_SCENARIOS,
    Scenario,
)

__all__ = ["InterpreterError", "infer_scenario_from_prompt"]

logger = logging.getLogger(__name__)

# Public fallback model + backend per #51 spec. Service deployments override
# these with PA_TUNE_*, PA_REFINE_*, or PA_VLM_* environment settings so
# prompt-only /tune uses the same internal model backend as the service.
_DEFAULT_CHAT_MODEL_NAME = "qwen/qwen3.5-397b-a17b"
_DEFAULT_CHAT_BACKEND = "nim"


class InterpreterError(RuntimeError):
    """Raised when the NL → Scenario interpreter cannot produce a valid Scenario."""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_DROP_SETTLE_RESTITUTION_EXAMPLE = {
    "name": "drop_settle",
    "metric": "settle_distance",
    "target": {
        "drop_height_m": 0.5,
        "duration_s": 2.0,
        "gravity": -9.81,
    },
    "parameters": [
        {"name": "restitution", "min": 0.4, "max": 0.95},
        {"name": "mass_scale", "min": 0.7, "max": 1.3},
    ],
}


def _render_target_prompt_lines() -> str:
    """Return the shared video-evidence target schema for both scenarios."""
    backend_names = " / ".join(f'"{name}"' for name in RENDERING_BACKEND_NAMES)
    return f"""    record_video  : "off" | "end_of_tune" | "always" (default "off").
                    Writes the trial's recording to PNG sequence + mp4 for
                    visual inspection without requiring a VLM call.
    video_renderer: backend name override for the render driver
                    ({backend_names}). Falls back to
                    vlm_renderer, then "ovrtx". ``mock`` creates deterministic
                    test evidence, not production visual evidence."""


_DROP_SETTLE_CONTACT_EXAMPLE = {
    "name": "drop_settle",
    "metric": "settle_distance",
    "target": {
        "drop_height_m": 0.5,
        "duration_s": 2.0,
        "gravity": -9.81,
    },
    "parameters": [
        {"name": "contact_ke", "min": 10000.0, "max": 100000.0},
        {"name": "contact_kd", "min": 0.0, "max": 1000.0},
        {"name": "mass_scale", "min": 0.7, "max": 1.3},
    ],
}

_FREEFORM_EXAMPLE = {
    "name": "freeform",
    "metric": "judge_score",
    "target": {
        "description": "spin a top on a smooth surface",
        "duration_s": 3.0,
        "gravity": -9.81,
        "initial_pose": {
            "position": [0.0, 0.5, 0.0],
            "rotation": [0.0, 0.0, 0.0],
        },
        "initial_velocity": [0.0, 0.0, 0.0],
        "initial_angular_velocity": [0.0, 30.0, 0.0],
        "surface": {"friction": 0.2},
        "observations": ["did the top fall over"],
    },
    "parameters": [
        {"name": "dynamic_friction", "min": 0.05, "max": 0.4},
        {"name": "mass_scale", "min": 0.5, "max": 1.5},
    ],
}


def _normalize_supported_param_keys(
    supported_param_keys: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """Validate and order an optional backend-specific parameter allowlist."""
    if supported_param_keys is None:
        return SUPPORTED_PARAM_KEYS
    raw = tuple(str(k) for k in supported_param_keys)
    invalid = sorted(set(raw) - set(SUPPORTED_PARAM_KEYS))
    if invalid:
        raise InterpreterError(
            "supported_param_keys contains unsupported parameter(s): "
            f"{invalid}. Supported globally: {sorted(SUPPORTED_PARAM_KEYS)}"
        )
    allowed = set(raw)
    ordered = tuple(k for k in SUPPORTED_PARAM_KEYS if k in allowed)
    if not ordered:
        raise InterpreterError("supported_param_keys must contain at least one key")
    return ordered


def _drop_settle_example_for_params(
    supported_param_keys: tuple[str, ...],
) -> dict[str, Any]:
    """Return a one-shot bouncy example that obeys the active backend."""
    if "restitution" in supported_param_keys:
        return _DROP_SETTLE_RESTITUTION_EXAMPLE
    if {"contact_ke", "contact_kd"}.issubset(set(supported_param_keys)):
        params = [
            {"name": "contact_ke", "min": 10000.0, "max": 100000.0},
            {"name": "contact_kd", "min": 0.0, "max": 1000.0},
        ]
        if "mass_scale" in supported_param_keys:
            params.append({"name": "mass_scale", "min": 0.7, "max": 1.3})
        return {**_DROP_SETTLE_CONTACT_EXAMPLE, "parameters": params}
    fallback_name = (
        "mass_scale"
        if "mass_scale" in supported_param_keys
        else supported_param_keys[0]
    )
    lo, hi = DEFAULT_PARAM_BOUNDS[fallback_name]
    return {
        **_DROP_SETTLE_RESTITUTION_EXAMPLE,
        "parameters": [{"name": fallback_name, "min": lo, "max": hi}],
    }


def _parameter_guidance(supported_param_keys: tuple[str, ...]) -> str:
    """Build parameter-selection examples that only mention legal keys."""
    lines: list[str] = []
    if "restitution" in supported_param_keys:
        lines.append('    "bouncy"        -> restitution biased high (e.g. 0.4..0.95)')
    elif {"contact_ke", "contact_kd"}.issubset(set(supported_param_keys)):
        lines.append(
            '    "bouncy"        -> contact_ke biased high and contact_kd biased '
            "low/moderate"
        )
    if "mass_scale" in supported_param_keys:
        lines.append('    "heavy"         -> mass_scale biased high (e.g. 1.2..2.0)')
    if {"static_friction", "dynamic_friction"}.issubset(set(supported_param_keys)):
        lines.extend(
            [
                '    "slippery"/"slick"/"icy"',
                "                    -> dynamic_friction + static_friction biased low",
                "                      (e.g. 0.05..0.3)",
                '    "sticky"/"grippy"',
                "                    -> static_friction biased high (e.g. 0.7..1.5)",
            ]
        )
    if "restitution" in supported_param_keys and {
        "static_friction",
        "dynamic_friction",
    }.issubset(set(supported_param_keys)):
        lines.append(
            '    "settle quickly"-> restitution biased low + friction biased high'
        )
    elif {"contact_ke", "contact_kd"}.issubset(set(supported_param_keys)):
        lines.append(
            '    "settle quickly"-> contact_kd biased higher and contact_ke moderated'
        )
    if not lines:
        lines.append("    use the most relevant parameter from the allowed set")
    return "\n".join(lines)


def _validate_scenario_param_allowlist(
    scenario: Scenario,
    supported_param_keys: tuple[str, ...],
) -> None:
    """Reject globally valid params that the active backend does not expose."""
    allowed = set(supported_param_keys)
    invalid = sorted({param.name for param in scenario.params} - allowed)
    if not invalid:
        return
    raise ScenarioParseError(
        "Scenario parameter(s) are not allowed for the active backend: "
        f"{', '.join(invalid)}. Allowed parameters: "
        f"{', '.join(sorted(supported_param_keys))}."
    )


def _validate_override_param_allowlist(
    scenario_override: dict[str, Any] | None,
    supported_param_keys: tuple[str, ...],
) -> None:
    """Reject explicit override params that the active backend cannot tune."""
    if not scenario_override:
        return
    raw_params = scenario_override.get("parameters")
    if not isinstance(raw_params, list):
        return
    allowed = set(supported_param_keys)
    names = {
        str(param.get("name"))
        for param in raw_params
        if isinstance(param, dict) and param.get("name") is not None
    }
    invalid = sorted(names - allowed)
    if not invalid:
        return
    raise InterpreterError(
        "scenario_override contains parameter(s) not allowed for the active "
        f"backend: {', '.join(invalid)}. Allowed parameters: "
        f"{', '.join(sorted(supported_param_keys))}."
    )


def _build_system_prompt(
    *,
    backend_name: str | None = None,
    supported_param_keys: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Build the system prompt that pins the JSON schema + 1-shot examples."""

    active_param_keys = _normalize_supported_param_keys(supported_param_keys)
    supported_scenarios = ", ".join(sorted(SUPPORTED_SCENARIOS))
    supported_params = ", ".join(sorted(active_param_keys))
    bounds_lines = "\n".join(
        f"      - {name}: [{lo}, {hi}]"
        for name, (lo, hi) in sorted(DEFAULT_PARAM_BOUNDS.items())
        if name in active_param_keys
    )
    backend_context = (
        f"\nActive backend: {backend_name}. Only choose tunable parameters from "
        "that backend's allowlist below.\n"
        if backend_name
        else ""
    )
    drop_settle_example = _drop_settle_example_for_params(active_param_keys)
    parameter_guidance = _parameter_guidance(active_param_keys)
    render_target_lines = _render_target_prompt_lines()

    return f"""You are a physics-tuning scenario interpreter. The user wants to
tune the physical-material parameters of a single rigid-body asset and has
described what they want in natural language. Your job is to translate that
description into a STRICT-JSON tuning scenario object.
{backend_context}

Respond with strict JSON only — no commentary, no markdown fences, no
preamble. The first character of your response MUST be ``{{`` and the last
character MUST be ``}}``.

# Schema

Top-level keys (all required unless noted):
    name        : one of [{supported_scenarios}]
    metric      : short string identifying the objective (e.g.
                  "settle_distance", "judge_score")
    target      : object — see per-kind keys below
    parameters  : non-empty list of {{name, min, max}} objects

Tunable parameter ``name`` MUST be one of: [{supported_params}].
Each parameter has numeric ``min`` and ``max`` (closed interval, min <= max).
If you are uncertain about a parameter's range, use these defaults:
{bounds_lines}

## drop_settle

Use ``drop_settle`` when the user describes an object-property adjective
(e.g. "bouncy", "heavy", "slippery", "sticky", "settle quickly"). It drops
the rigid body from a fixed height onto a flat plane and measures how it
settles. ``target`` recognises the following keys (all optional —
omit ones the prompt does not constrain):
    drop_height_m : float, the GAP (in meters) between the body's
                    bottom and the ground. Default = the body's own
                    height (so the body sits one own-height above the
                    ground before dropping). NOT an absolute Y position.
    duration_s    : float, simulated duration in seconds (default ~2.0)
    gravity       : float, signed gravity in m/s^2 (default -9.81)
    sample_fps    : int, recording frame rate (default 30, max 60).
                    Drives how dense recording.usd time samples are.
    cameras       : list of camera directions (default ["+x+y+z"], a
                    tilted corner view that shows all three axes) for
                    the scale-aware cameras authored on the scene.
                    Cardinal directions ("+x", "-z", ...) author
                    orthographic-axis side views; corner triples
                    ("+x+y+z", "-x-y-z", "+x-0.5y+z") author tilted
                    isometric views. Corner directions work the same
                    on Y-up and Z-up assets, which is why the default
                    is a corner.
    vlm_check     : "off" | "end_of_tune" | "always" (default "off").
                    Optional VLM verdict on the final-state render.
                    Never replaces the programmatic settle_distance
                    objective; only attaches a verdict for audit.
{render_target_lines}

## freeform

Use ``freeform`` when the prompt implies non-default INITIAL CONDITIONS — a
specific pose, linear velocity, angular velocity, or surface friction
(e.g. "spin a top", "drop sideways", "throw it forward", "slide on ice").
``target`` keys (all optional except ``description``):
    description              : short paraphrase of the user's intent
    duration_s               : float, seconds
    gravity                  : float, signed m/s^2
    initial_pose             : {{position: [x, y, z], rotation: [rx, ry, rz]}}
                               (rotation is XYZ Euler in radians)
    initial_velocity         : [vx, vy, vz] (m/s)
    initial_angular_velocity : [wx, wy, wz] (rad/s)
    surface                  : {{friction: float}}
    observations             : list of short natural-language strings
                               (consumed by the hybrid programmatic
                                score AND the VLM judge)
    sample_fps               : int, recording frame rate (default 30,
                               max 60). Drives recording.usd density.
    cameras                  : list of camera directions (default
                               ["+x+y+z"], a tilted corner view).
                               Cardinal directions author side views;
                               corner triples author tilted isometric
                               views. Corner is the default because it
                               works on both Y-up and Z-up assets.
{render_target_lines}

When uncertain between drop_settle and freeform, prefer drop_settle — it is
the validated path with a fixed target schema.

# Parameter selection

Pick the smallest set of parameters that the user's prompt actually
constrains, and TIGHTEN their bounds around the user's intent. Examples:
{parameter_guidance}
At minimum, output one parameter. If the prompt is silent on physics
properties, default to ``mass_scale`` over its full default range when
``mass_scale`` is available; otherwise choose the broadest relevant allowed
parameter.

# Examples

Example 1 — input: "make this object bouncy":
{json.dumps(drop_settle_example, indent=2)}

Example 2 — input: "spin a top on a smooth surface":
{json.dumps(_FREEFORM_EXAMPLE, indent=2)}
"""


def _build_user_prompt(
    user_prompt: str,
    *,
    physics_usd: Path | None,
    backend_name: str | None,
    supported_param_keys: tuple[str, ...],
    retry_error: str | None,
) -> str:
    """Build the per-call user prompt — text + optional USD context + retry hint."""
    parts: list[str] = []
    if physics_usd is not None:
        # Surface only the basename for context — full paths leak workspace
        # layout into the LLM and add no signal.
        parts.append(f"Asset (context only): {Path(physics_usd).name}")
    if backend_name:
        parts.append(f"Active backend: {backend_name}")
    parts.append(
        "Allowed tunable parameters for this backend: "
        + ", ".join(sorted(supported_param_keys))
    )
    parts.append(f"User prompt: {user_prompt.strip()}")
    if retry_error:
        parts.append(
            "Your previous response failed schema validation with this error:\n"
            f"    {retry_error}\n"
            "Fix the issue and respond again with strict JSON only."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction + merging helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any]:
    """Defensively extract the first balanced JSON object from ``text``.

    Handles preambles like ``"Here is the JSON: { ... }"`` and code-fence
    wrappers like ``"```json\\n{...}\\n```"``. Returns the parsed dict.

    Raises:
        InterpreterError: if no balanced JSON object can be located or the
            located substring fails ``json.loads``.
    """
    if not isinstance(text, str) or not text.strip():
        raise InterpreterError("LLM returned an empty response")

    start = text.find("{")
    if start == -1:
        raise InterpreterError(f"LLM response contained no JSON object: {text[:200]!r}")

    # Walk the string tracking string literals + escapes so braces inside
    # quoted strings do not throw off the depth counter.
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise InterpreterError(
            "LLM response contained an unbalanced JSON object — could not "
            "locate matching closing brace"
        )

    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as e:
        raise InterpreterError(
            f"LLM response was not valid JSON: {e}; snippet={snippet[:200]!r}"
        ) from e
    if not isinstance(parsed, dict):
        raise InterpreterError(
            f"LLM JSON must be an object, got {type(parsed).__name__}"
        )
    return parsed


def _merge_explicit_wins(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` — explicit fields win.

    Semantics:
        * Scalar / non-dict values in ``override`` replace ``base`` outright.
        * Nested dicts merge recursively.
        * Lists in ``override`` replace lists in ``base`` wholesale (no
          element-wise merging — list shapes can differ in arity / order).
        * Keys present only in ``base`` are preserved.
        * Keys present only in ``override`` are added.

    The function is pure — neither input dict is mutated.
    """
    result: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            result[key] = _merge_explicit_wins(result[key], override_value)
        else:
            # Includes the list-replacement case and any scalar override.
            result[key] = override_value
    return result


# ---------------------------------------------------------------------------
# Default chat model resolution
# ---------------------------------------------------------------------------


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _default_chat_backend_and_model() -> tuple[str, str | None]:
    backend = _first_env("PA_TUNE_BACKEND", "PA_REFINE_BACKEND", "PA_VLM_BACKEND")
    if not backend:
        backend = _DEFAULT_CHAT_BACKEND

    model = _first_env("PA_TUNE_MODEL", "PA_REFINE_MODEL", "PA_VLM_MODEL")
    if not model and backend == _DEFAULT_CHAT_BACKEND:
        model = _DEFAULT_CHAT_MODEL_NAME
    return backend, model


def _resolve_default_chat_model() -> Any:
    """Resolve the default chat model.

    Mirrors the wiring style used by other agents (see
    ``apps/material_agent/material_agent/scene/llm_refine.py``): import the
    factory lazily so this module remains cheap to import when the caller
    injects ``chat_model`` directly (e.g. tests).

    Raises:
        InterpreterError: when no chat-model factory is reachable or none of
            the project's chat factories can be instantiated.
    """
    try:
        from world_understanding.functions.models.chat_models import (
            create_chat_model_from_config,
        )
    except ImportError as e:  # pragma: no cover — defensive
        raise InterpreterError(
            "Default chat model factory is unavailable; pass chat_model "
            "explicitly to infer_scenario_from_prompt()."
        ) from e

    backend, model = _default_chat_backend_and_model()
    llm_config: dict[str, Any] = {
        "backend": backend,
        "temperature": float(os.getenv("PA_TUNE_TEMPERATURE", "0.0")),
        "max_tokens": int(os.getenv("PA_TUNE_MAX_TOKENS", "4096")),
    }
    if model:
        llm_config["model"] = model
    base_url = _first_env("PA_TUNE_BASE_URL", "PA_REFINE_BASE_URL", "PA_VLM_BASE_URL")
    api_key_env = _first_env(
        "PA_TUNE_API_KEY_ENV",
        "PA_REFINE_API_KEY_ENV",
        "PA_VLM_API_KEY_ENV",
    )
    api_key = _first_env("PA_TUNE_API_KEY", "PA_REFINE_API_KEY", "PA_VLM_API_KEY")
    if base_url:
        llm_config["base_url"] = base_url
    if api_key_env:
        llm_config["api_key_env"] = api_key_env
    elif api_key:
        llm_config["api_key"] = api_key

    try:
        chat_model = create_chat_model_from_config(llm_config)
        if chat_model is None:
            raise InterpreterError(
                f"No API key available for tune scenario interpreter backend {backend!r}."
            )
        return chat_model
    except Exception as e:
        if isinstance(e, InterpreterError):
            raise
        # Codex round 4: default-model construction can raise provider
        # exceptions carrying endpoints, deployment names, or auth
        # config. Log the verbose form server-side only and raise a
        # stable, redacted error so nothing crosses the REST boundary.
        logger.warning(
            "default chat-model factory raised (provider detail logged "
            "server-side only): %s",
            e,
        )
        raise InterpreterError(
            "Failed to instantiate default chat model for backend "
            f"{backend!r}; see server logs for provider detail."
        ) from e


# ---------------------------------------------------------------------------
# Audit (write-only — never read back to skip an LLM call)
# ---------------------------------------------------------------------------


def _write_audit_record(
    audit_dir: Path,
    final_dict: dict[str, Any],
    *,
    user_prompt: str,
    model_name: str,
    merged_from_explicit: bool,
) -> None:
    """Atomically write ``inferred_scenario.json`` into ``audit_dir``.

    Write-only — there is no companion read function. The file is an
    audit trail of what the LLM authored on each tune run, NOT a cache
    that's consulted to skip subsequent LLM calls. Re-running with the
    same inputs will overwrite this file with a fresh LLM-authored
    scenario.

    The write is atomic: contents go to ``inferred_scenario.json.tmp``
    first, then ``os.replace`` swaps them into place so a crash
    mid-write cannot leave the file half-populated.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    target = audit_dir / "inferred_scenario.json"
    tmp = audit_dir / "inferred_scenario.json.tmp"

    payload: dict[str, Any] = {
        "_meta": {
            "user_prompt": user_prompt,
            "model": model_name,
            "merged_from_explicit": merged_from_explicit,
        }
    }
    payload.update(final_dict)

    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    os.replace(tmp, target)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _call_llm(chat_model: Any, system_prompt: str, user_prompt: str) -> str:
    """Issue one LLM call and return the response text.

    Wraps :func:`generate_chat_response` so error handling stays uniform —
    that helper returns ``{"error": ...}`` on failure rather than raising.

    **Error sanitization (codex round-3 fix):** the provider's exception
    string can include internal endpoints, deployment names, request IDs,
    or auth/config metadata. Those leak through ``InterpreterError`` ->
    runner -> session metadata -> ``GET /tune/{id}/results`` REST
    response. We log the full provider error at WARNING level for
    operator debugging and raise a stable, generic message across the
    boundary.
    """
    result = generate_chat_response(
        chat_model=chat_model,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )
    if "error" in result:
        # Server-side observability: keep the verbose provider error in
        # the application log under our control.
        logger.warning(
            "interpreter LLM call failed (provider detail logged server-side only): %s",
            result["error"],
        )
        # User-visible: a stable, redacted message.
        raise InterpreterError("LLM call failed; see server logs for provider detail.")
    response = result.get("response", "")
    if not isinstance(response, str):
        raise InterpreterError(
            f"LLM response was not a string: {type(response).__name__}"
        )
    return response


def infer_scenario_from_prompt(
    user_prompt: str,
    *,
    scenario_override: dict[str, Any] | None = None,
    chat_model: Any | None = None,
    audit_dir: Path | None = None,
    physics_usd: Path | None = None,
    backend_name: str | None = None,
    supported_param_keys: tuple[str, ...] | list[str] | None = None,
) -> Scenario:
    """Translate a natural-language prompt into a validated :class:`Scenario`.

    No caching: every call reaches the LLM. The optional ``audit_dir``
    is write-only — when supplied, the post-merge final dict (the one
    passed to :func:`parse_scenario`) is atomically written to
    ``audit_dir/inferred_scenario.json`` so a human reviewer can audit
    what the LLM authored. The file is NOT read back to skip
    subsequent LLM calls.

    Args:
        user_prompt: Free-form natural-language description (e.g.
            ``"make this object bouncy"``). Must be a non-empty string.
        scenario_override: Optional dict of explicit user YAML / config that
            wins on every key conflict. The interpreter only fills gaps the
            user's config did not specify. Must be a dict if provided.
        chat_model: Optional pre-built chat model instance. When ``None`` the
            project default (``qwen/qwen3.5-397b-a17b`` on the ``nim``
            backend) is resolved via
            :func:`world_understanding.functions.models.chat_models.create_chat_model`.
        audit_dir: Optional directory. When set, the post-merge final dict
            is written to ``audit_dir/inferred_scenario.json`` for audit
            (write-only — never read back).
        physics_usd: Optional path to the simulation-ready USD. Only the
            basename is surfaced into the LLM prompt — full paths leak
            workspace layout without adding signal.
        backend_name: Optional active simulation backend name, used only to
            steer the LLM prompt.
        supported_param_keys: Optional backend-specific parameter allowlist.
            When provided, the prompt instructs the LLM to emit only these
            parameter names.

    Returns:
        A validated :class:`Scenario` (kind ``drop_settle`` or ``freeform``).

    Raises:
        InterpreterError: when the LLM call fails, the response cannot be
            parsed as JSON, or the merged dict still fails schema validation
            on the second attempt. The underlying parser error is chained.
    """
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise InterpreterError("user_prompt must be a non-empty string")
    if scenario_override is not None and not isinstance(scenario_override, dict):
        raise InterpreterError(
            f"scenario_override must be a dict or None, got "
            f"{type(scenario_override).__name__}"
        )

    if chat_model is None:
        default_backend, default_model = _default_chat_backend_and_model()
        chat_model = _resolve_default_chat_model()
        model_name = (
            getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or f"{default_backend}/{default_model}"
        )
    else:
        # Best-effort introspection — only used for the cache _meta block.
        model_name = (
            getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or type(chat_model).__name__
        )

    active_param_keys = _normalize_supported_param_keys(supported_param_keys)
    _validate_override_param_allowlist(scenario_override, active_param_keys)
    system_prompt = _build_system_prompt(
        backend_name=backend_name,
        supported_param_keys=active_param_keys,
    )

    # ---- Attempt 1: bare prompt -----------------------------------------
    last_error: Exception | None = None
    final_dict: dict[str, Any] | None = None
    parser_error_text: str | None = None

    for attempt in range(2):
        user_message = _build_user_prompt(
            user_prompt,
            physics_usd=physics_usd,
            backend_name=backend_name,
            supported_param_keys=active_param_keys,
            retry_error=parser_error_text if attempt > 0 else None,
        )
        try:
            raw_response = _call_llm(chat_model, system_prompt, user_message)
            llm_dict = _extract_json(raw_response)
        except InterpreterError as e:
            # JSON-extraction / LLM-call problems on attempt 0 deserve a
            # retry too — feed the error back as the retry hint.
            last_error = e
            parser_error_text = str(e)
            logger.warning(
                "Interpreter attempt %d failed during LLM/JSON extraction: %s",
                attempt + 1,
                e,
            )
            continue

        # Apply the user's explicit override AFTER the LLM produces its
        # candidate dict but BEFORE schema validation, so the validator sees
        # the same object that gets cached and downstream consumers get.
        #
        # Codex round 10 cross-kind leak fix: when the explicit override
        # specifies a different ``name`` than the LLM authored (e.g. user
        # forces drop_settle but the LLM emitted freeform), drop the
        # LLM's kind-specific fields wholesale before merging. Otherwise
        # an explicit drop_settle YAML that omits optional ``metric``
        # could inherit freeform's ``metric: judge_score`` and unrelated
        # freeform target keys (description, initial_pose, …), passing
        # the parser but breaking the ovphysx adapter at runtime.
        if scenario_override:
            override_name = scenario_override.get("name")
            llm_name = llm_dict.get("name")
            if (
                isinstance(override_name, str)
                and isinstance(llm_name, str)
                and override_name != llm_name
            ):
                # Different scenario kind explicitly requested: keep only
                # the override's structure, augment with bound-defaults
                # from the parser, and ignore LLM-authored fields.
                # CodeRabbit thread 5_4NnQ: when the cross-kind override
                # carries no ``parameters``, the LLM's params can't be
                # imported either (they were authored for a different
                # scenario kind and may reference incompatible bounds),
                # so this branch produces a dict that fails parse_scenario
                # for missing ``parameters``. Surface a clear actionable
                # error instead of bouncing through the retry loop and
                # falling out as an opaque InterpreterError.
                if "parameters" not in scenario_override:
                    raise InterpreterError(
                        "scenario_override forced a different scenario kind "
                        f"({override_name!r}) than the LLM inferred "
                        f"({llm_name!r}), but did not supply 'parameters'. "
                        "Cross-kind LLM-authored params can't be reused: "
                        "include 'parameters' in scenario_override (with "
                        "appropriate bounds for the override's scenario "
                        "kind), or drop the explicit 'name' override and "
                        "let the LLM author both the kind and parameters."
                    )
                merged = dict(scenario_override)
            else:
                merged = _merge_explicit_wins(llm_dict, scenario_override)
        else:
            merged = llm_dict

        try:
            scenario = parse_scenario(merged)
            _validate_scenario_param_allowlist(scenario, active_param_keys)
        except ScenarioParseError as e:
            last_error = e
            parser_error_text = str(e)
            logger.warning(
                "Interpreter attempt %d produced a dict that failed parse_scenario: %s",
                attempt + 1,
                e,
            )
            continue

        final_dict = merged
        break
    else:
        # for-else: both attempts exhausted without break.
        assert last_error is not None
        raise InterpreterError(
            f"Could not produce a valid Scenario after 2 attempts: {last_error}"
        ) from last_error

    assert final_dict is not None  # narrow for mypy

    if audit_dir is not None:
        try:
            _write_audit_record(
                audit_dir,
                final_dict,
                user_prompt=user_prompt,
                model_name=str(model_name),
                merged_from_explicit=bool(scenario_override),
            )
        except (OSError, TypeError, ValueError) as e:
            # Audit is best-effort; surface but don't fail the run if
            # disk is full / permissions are off (OSError) or the
            # scenario carries a non-JSON-serializable value
            # (TypeError/ValueError from json.dumps). The caller already
            # has the valid Scenario in hand.
            logger.warning(
                "Failed to write inferred_scenario.json audit record to %s: %s",
                audit_dir,
                e,
            )

    return scenario
