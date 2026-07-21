# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a compact Workbench material-authoring run packet."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from content_workbench_agent_client import (
    close_session as _client_close_session,
)
from content_workbench_agent_client import (
    create_session as _client_create_session,
)
from content_workbench_agent_client import (
    download_agent_api_docs as _client_download_agent_api_docs,
)
from content_workbench_agent_client import (
    download_to_file as _client_download_to_file,
)
from content_workbench_agent_client import (
    get_optional_json as _fetch_optional_json,
)
from content_workbench_agent_client import (
    post_json as _client_post_json,
)
from content_workbench_agent_client import (
    render_view as _client_render_view,
)

from content_workflow_cli.trace import append_jsonl, utc_now

from .snapshot_scene import (
    MaterialCandidatePolicy,
    fetch_snapshot,
    write_snapshot_artifacts,
)

PACKET_SCHEMA_VERSION = "content-agents.material-run-packet.v1"
DEFAULT_PACKET_RENDER_VIEWS = (
    {"name": "initial_top", "direction": "+z"},
    {"name": "initial_bottom", "direction": "-z"},
    {"name": "initial_oblique", "direction": "+x-y+z"},
)


@dataclass(frozen=True)
class MaterialRunPacketConfig:
    workbench_url: str
    run_dir: Path
    usd_path: Path
    materials_yaml: Path
    materials_usd: Path
    optimize: bool = True
    respect_existing_material_bindings: bool = False
    root_prim_path: str | None = None
    material_candidate_space: str = "source"
    skip_instances: bool = True
    skip_prototypes: bool = False
    skip_invisible: bool = False
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = None
    width: int = 640
    height: int = 480
    render_quality: str = "inspection"


def prepare_material_run_packet(config: MaterialRunPacketConfig) -> dict[str, Any]:
    """Create a Workbench session, snapshot context, and initial evidence renders."""

    run_dir = config.run_dir
    raw_dir = run_dir / "raw"
    evidence_dir = run_dir / "evidence_renders"
    trace_dir = run_dir / "trace"
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    workbench_url = config.workbench_url.rstrip("/")
    docs = _fetch_docs(workbench_url, raw_dir)
    session = _create_session(config)
    session_id = str(session["session_id"])
    (raw_dir / "session_create.json").write_text(
        json.dumps(session, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    try:
        encoded_session_id = quote(session_id, safe="")
        optimization = _fetch_optional_json(
            f"{workbench_url}/sessions/{encoded_session_id}/optimization"
        )
        if optimization is not None:
            (raw_dir / "optimization.json").write_text(
                json.dumps(optimization, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        snapshot = fetch_snapshot(
            workbench_url=workbench_url,
            session_id=session_id,
            root_prim_path=config.root_prim_path,
            timeout=300.0,
        )
        candidate_policy = MaterialCandidatePolicy(
            material_candidate_space=config.material_candidate_space,
            root_prim_path=config.root_prim_path,
            skip_instances=config.skip_instances,
            skip_prototypes=config.skip_prototypes,
            skip_invisible=config.skip_invisible,
        )
        snapshot_artifacts = write_snapshot_artifacts(
            snapshot,
            run_dir,
            materials_yaml=config.materials_yaml,
            materials_usd=config.materials_usd,
            append_trace=True,
            respect_existing_material_bindings=config.respect_existing_material_bindings,
            candidate_policy=candidate_policy,
        )
        initial_renders = [
            _render_view(
                workbench_url=workbench_url,
                session_id=session_id,
                output_dir=evidence_dir,
                name=str(view["name"]),
                direction=str(view["direction"]),
                width=config.width,
                height=config.height,
                render_quality=config.render_quality,
            )
            for view in DEFAULT_PACKET_RENDER_VIEWS
        ]
        initial_render_records = raw_dir / "initial_render_records.json"
        initial_render_records.write_text(
            json.dumps(initial_renders, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        render_downloads = sum(
            int(record.get("artifact_download_count") or 0)
            for record in initial_renders
            if isinstance(record, dict)
        )

        packet = {
            "schema_version": PACKET_SCHEMA_VERSION,
            "created_at": utc_now(),
            "workbench_url": workbench_url,
            "session_id": session_id,
            "source_usd": str(config.usd_path),
            "materials_yaml": str(config.materials_yaml),
            "materials_usd": str(config.materials_usd),
            "optimize": config.optimize,
            "optimizer_options": {
                "flatten_prototypes": config.flatten_prototypes,
                "enable_deinstance": config.enable_deinstance,
                "enable_split": config.enable_split,
                "enable_deduplicate": config.enable_deduplicate,
            },
            "material_candidate_policy": candidate_policy.as_dict(),
            "clear_materials": not config.respect_existing_material_bindings,
            "respect_existing_material_bindings": (
                config.respect_existing_material_bindings
            ),
            "docs": docs,
            "session": {
                "session_create": str(raw_dir / "session_create.json"),
                "optimization": str(raw_dir / "optimization.json")
                if optimization is not None
                else None,
            },
            "snapshot": snapshot_artifacts,
            "initial_evidence_renders": initial_renders,
            "recommended_normal_path": [
                (
                    "Use the attached reference and initial Workbench render images "
                    "for the first visual pass."
                ),
                (
                    "Read material_authoring_context.md and "
                    "material_assignment_seed.json only when path/material detail is "
                    "needed."
                ),
                (
                    "Assign library materials for every canonical material candidate "
                    "family. The Workbench session starts from clean materials unless "
                    "existing bindings were explicitly requested."
                ),
                (
                    "Use targeted Workbench calls only for unresolved ambiguity; do "
                    "not recreate docs, session, snapshot, or initial renders."
                ),
            ],
            "operation_counts_so_far": {
                "workbench_api_calls_total": 3
                + 1
                + (1 if optimization is not None else 0)
                + 1
                + len(initial_renders)
                + render_downloads,
                "docs_fetches": 3,
                "session_creates": 1,
                "optimization_reads": 1 if optimization is not None else 0,
                "scene_snapshots": 1,
                "render_calls_total": len(initial_renders),
                "render_artifact_downloads": render_downloads,
                "pick_calls": 0,
                "command_calls": 0,
            },
        }
        packet_path = raw_dir / "material_run_packet.json"
        packet_path.write_text(
            json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8"
        )
        _append_packet_trace(run_dir, packet_path, packet)
        return packet
    except Exception:
        try:
            _delete_session(workbench_url, session_id)
        except Exception as close_exc:  # noqa: BLE001 - preserve original failure
            (raw_dir / "session_close_error.json").write_text(
                json.dumps(
                    {
                        "session_id": session_id,
                        "error_type": type(close_exc).__name__,
                        "error": str(close_exc),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        raise


def packet_image_inputs(packet: dict[str, Any]) -> list[dict[str, str]]:
    """Return image attachments suitable for a child model request."""

    images = []
    for record in packet.get("initial_evidence_renders", []):
        if not isinstance(record, dict):
            continue
        image_path = record.get("image_path")
        label = record.get("name")
        if (
            isinstance(image_path, str)
            and isinstance(label, str)
            and Path(image_path).is_file()
        ):
            images.append(
                {
                    "label": f"Workbench initial render: {label}",
                    "path": image_path,
                }
            )
    return images


def _fetch_docs(workbench_url: str, raw_dir: Path) -> dict[str, str]:
    return _client_download_agent_api_docs(workbench_url, raw_dir)


def _post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    return _client_post_json(url, body)


def _download_to_file(url: str, path: Path) -> None:
    _client_download_to_file(url, path)


def _create_session(config: MaterialRunPacketConfig) -> dict[str, Any]:
    payload = {
        "scene_path": str(config.usd_path),
        "optimize": config.optimize,
        "clear_materials": not config.respect_existing_material_bindings,
        "width": config.width,
        "height": config.height,
    }
    for key in (
        "flatten_prototypes",
        "enable_deinstance",
        "enable_split",
        "enable_deduplicate",
    ):
        value = getattr(config, key)
        if value is not None:
            payload[key] = value
    return _client_create_session(config.workbench_url, payload)


def _delete_session(workbench_url: str, session_id: str) -> None:
    _client_close_session(workbench_url, session_id)


def _render_view(
    *,
    workbench_url: str,
    session_id: str,
    output_dir: Path,
    name: str,
    direction: str,
    width: int,
    height: int,
    render_quality: str,
) -> dict[str, Any]:
    return _client_render_view(
        workbench_url=workbench_url,
        session_id=session_id,
        output_dir=output_dir,
        name=name,
        direction=direction,
        width=width,
        height=height,
        render_quality=render_quality,
    )


def _append_packet_trace(
    run_dir: Path, packet_path: Path, packet: dict[str, Any]
) -> None:
    append_jsonl(
        run_dir / "trace" / "events.jsonl",
        {
            "schema_version": "content-agents.trace.v1",
            "time": utc_now(),
            "event_type": "api",
            "phase": "material_run_packet",
            "summary": (
                "Prepared reusable material workflow packet with Workbench docs, "
                "session, compact scene snapshot, and initial evidence renders."
            ),
            "artifacts": [
                str(packet_path),
                *[
                    str(record.get("image_path"))
                    for record in packet.get("initial_evidence_renders", [])
                    if isinstance(record, dict) and record.get("image_path")
                ],
            ],
            "data": {
                "session_id": packet.get("session_id"),
                "api_calls": [
                    "GET /agent-api",
                    "GET /agent-api.json",
                    "GET /openapi.json",
                    "POST /sessions",
                    "GET /sessions/{session_id}/optimization",
                    "POST /sessions/{session_id}/scene/snapshot",
                    "POST /sessions/{session_id}/render",
                ],
                "operation_counts_so_far": packet.get("operation_counts_so_far"),
            },
        },
    )
