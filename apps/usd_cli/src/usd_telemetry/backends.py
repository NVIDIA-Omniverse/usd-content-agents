# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Telemetry backends: file (JSONL, default), otlp (OTLP/HTTP JSON), none.

Backends receive the OTel-shaped record dict from span.build_record(). A
backend failure must never affect the wrapped command — emit_all() swallows
everything (printing to stderr only under USD_CLI_TEL_DEBUG=1).

No third-party dependencies: the OTLP backend hand-rolls the OTLP/HTTP JSON
encoding of ExportTraceServiceRequest via urllib.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from .config import Config


def _json_default(obj: object) -> str:
    return str(obj)


def _dumps(record: dict) -> str:
    return json.dumps(
        record, separators=(",", ":"), default=_json_default, ensure_ascii=False
    )


# ---------------------------------------------------------------- file (JSONL)


def _rotate_if_needed(cfg: Config) -> None:
    if cfg.file_max_bytes <= 0:
        return
    try:
        if cfg.log_path.stat().st_size >= cfg.file_max_bytes:
            rotated = cfg.log_path.with_name(cfg.log_path.name + ".1")
            cfg.log_path.replace(rotated)
    except OSError:
        pass


def emit_file(record: dict, cfg: Config) -> None:
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(cfg)
    line = _dumps(record) + "\n"
    with open(cfg.log_path, "a", encoding="utf-8") as fh:
        fh.write(line)


# ---------------------------------------------------------------- otlp (HTTP)


def _otlp_any_value(value: object) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_any_value(v) for v in value]}}
    return {"stringValue": str(value)}


def _otlp_attrs(attrs: dict) -> list[dict]:
    return [{"key": k, "value": _otlp_any_value(v)} for k, v in attrs.items()]


def to_otlp_payload(record: dict) -> dict:
    """Map one record to an OTLP/HTTP JSON ExportTraceServiceRequest."""
    status_code = 1 if record["status"].get("code") == "OK" else 2
    status: dict[str, object] = {"code": status_code}
    if "message" in record["status"]:
        status["message"] = record["status"]["message"]

    span = {
        "traceId": record["trace_id"],
        "spanId": record["span_id"],
        "name": record["name"],
        "kind": 3,  # SPAN_KIND_CLIENT
        "startTimeUnixNano": str(record["start_time_unix_nano"]),
        "endTimeUnixNano": str(record["end_time_unix_nano"]),
        "attributes": _otlp_attrs(record["attributes"]),
        "status": status,
    }
    if record.get("parent_span_id"):
        span["parentSpanId"] = record["parent_span_id"]
    if record.get("trace_state"):
        span["traceState"] = record["trace_state"]
    sdk_version = record["attributes"].get("telemetry.sdk.version", "")
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _otlp_attrs({"service.name": "usd-cli"})
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "usd-cli-tel", "version": str(sdk_version)},
                        "spans": [span],
                    }
                ],
            }
        ]
    }


def emit_otlp(record: dict, cfg: Config) -> None:
    body = _dumps(to_otlp_payload(record)).encode("utf-8")
    req = urllib.request.Request(
        cfg.otlp_endpoint,
        data=body,
        headers={"Content-Type": "application/json", **cfg.otlp_headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg.otlp_timeout) as resp:
        resp.read()


# ------------------------------------------------------------------- registry

_BACKENDS = {
    "file": emit_file,
    "otlp": emit_otlp,
    "none": lambda record, cfg: None,
}


def emit_all(record: dict, cfg: Config) -> None:
    for name in cfg.backends:
        emitter = _BACKENDS.get(name)
        try:
            if emitter is None:
                raise ValueError(f"unknown telemetry backend: {name!r}")
            emitter(record, cfg)
        except Exception as exc:  # never let telemetry break the wrapped call
            if cfg.debug:
                print(f"usd-cli-tel: backend {name!r} failed: {exc}", file=sys.stderr)
