# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded command-line access to agent-side observation memory."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import ValidationError

from .memory import AgentMemory, MemoryError, MemorySearchQuery, RememberRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="content-agent-memory",
        description="Record and retrieve durable observations for one agent run.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--memory-root", type=Path)
    parser.add_argument(
        "--broker-url",
        help=(
            "Wrapper-owned loopback memory broker. Mesh workflows use this "
            "instead of granting the child direct access to the memory store."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("init", help="Initialize a memory run.")

    record = commands.add_parser("record", help="Record one observation.")
    record.add_argument("--input", type=Path, required=True)

    context = commands.add_parser("context", help="Get bounded working context.")
    context.add_argument("--limit", type=int, default=12)

    search = commands.add_parser("search", help="Search current-run observations.")
    search.add_argument("--text")
    search.add_argument("--workflow")
    search.add_argument("--phase")
    search.add_argument("--scene-revision-id")
    search.add_argument("--operation")
    search.add_argument(
        "--outcome",
        choices=("matched", "contradicted", "ambiguous", "not_checked"),
    )
    search.add_argument("--importance", choices=("low", "normal", "high", "critical"))
    search.add_argument("--tag")
    search.add_argument("--target")
    search.add_argument("--artifact-role")
    search.add_argument("--sequence-min", type=int)
    search.add_argument("--sequence-max", type=int)
    search.add_argument("--limit", type=int, default=12)

    inspect = commands.add_parser(
        "inspect",
        help=("Inspect records; --artifact-role is required to materialize evidence."),
    )
    inspect.add_argument("--observation-id", action="append", required=True)
    inspect.add_argument("--artifact-role", action="append", default=[])
    inspect.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)

    for command in ("pin", "unpin"):
        mutation = commands.add_parser(
            command, help=f"{command.title()} an observation."
        )
        mutation.add_argument("--observation-id", required=True)

    commands.add_parser("rebuild", help="Rebuild SQLite from the canonical journal.")
    commands.add_parser(
        "recover-torn-final-line",
        help="Preserve and remove an incomplete final journal line.",
    )
    return parser


def _read_request(path: Path) -> RememberRequest:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return RememberRequest.model_validate(payload)


def _emit(payload: Any) -> None:
    if hasattr(payload, "model_dump_json"):
        print(payload.model_dump_json(indent=2))
        return
    if isinstance(payload, tuple):
        print(
            json.dumps(
                [item.model_dump(mode="json") for item in payload],
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def _broker_request(
    broker_url: str,
    *,
    run_id: str,
    command: str,
    payload: dict[str, Any],
) -> Any:
    parsed = urlparse(broker_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or not parsed.port
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--broker-url must be a root HTTP loopback URL with a port")
    try:
        response = requests.post(
            f"{broker_url.rstrip('/')}/v1/{command}",
            json={"run_id": run_id, **payload},
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        raise ValueError(f"memory broker request failed: {exc}") from exc
    if not isinstance(result, dict | list):
        raise ValueError("memory broker returned a non-JSON result")
    return result


def _materialize_broker_inspect(payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise ValueError("memory broker inspect returned an invalid result")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        return payload
    lease_id = payload.get("lease_id")
    if not isinstance(lease_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", lease_id):
        raise ValueError("memory broker inspect returned an invalid lease_id")
    lease_dir = Path.cwd() / ".agent-memory-inspect" / lease_id
    lease_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    materialized: list[dict[str, Any]] = []
    try:
        for index, raw_artifact in enumerate(raw_artifacts):
            if not isinstance(raw_artifact, dict):
                raise ValueError("memory broker inspect returned invalid artifact data")
            encoded = raw_artifact.get("content_base64")
            filename = raw_artifact.get("filename")
            expected_digest = raw_artifact.get("sha256")
            if (
                not isinstance(encoded, str)
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not re.fullmatch(r"[A-Za-z0-9._-]+", filename)
                or not isinstance(expected_digest, str)
            ):
                raise ValueError("memory broker inspect returned invalid artifact data")
            try:
                content = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError(
                    "memory broker inspect returned invalid base64"
                ) from exc
            if hashlib.sha256(content).hexdigest() != expected_digest:
                raise ValueError(
                    "memory broker inspect artifact failed digest verification"
                )
            target = lease_dir / f"{index:02d}-{filename}"
            with target.open("xb") as stream:
                stream.write(content)
            artifact = dict(raw_artifact)
            artifact.pop("content_base64", None)
            artifact.pop("filename", None)
            artifact["path"] = str(target)
            materialized.append(artifact)
    except Exception:
        for candidate in lease_dir.iterdir():
            if candidate.is_file() and not candidate.is_symlink():
                candidate.unlink()
        lease_dir.rmdir()
        raise
    return {**payload, "artifacts": materialized}


def _dispatch_broker(args: argparse.Namespace) -> None:
    if args.command == "init":
        payload = _broker_request(
            args.broker_url, run_id=args.run_id, command="init", payload={}
        )
    elif args.command == "record":
        request = _read_request(args.input)
        payload = _broker_request(
            args.broker_url,
            run_id=args.run_id,
            command="record",
            payload={"request": request.model_dump(mode="json")},
        )
    elif args.command == "context":
        payload = _broker_request(
            args.broker_url,
            run_id=args.run_id,
            command="context",
            payload={"limit": args.limit},
        )
    elif args.command == "search":
        query = MemorySearchQuery(
            text=args.text,
            workflow=args.workflow,
            phase=args.phase,
            scene_revision_id=args.scene_revision_id,
            operation=args.operation,
            outcome=args.outcome,
            importance=args.importance,
            tag=args.tag,
            target=args.target,
            artifact_role=args.artifact_role,
            sequence_min=args.sequence_min,
            sequence_max=args.sequence_max,
            limit=args.limit,
        )
        payload = _broker_request(
            args.broker_url,
            run_id=args.run_id,
            command="search",
            payload={"query": query.model_dump(mode="json")},
        )
    elif args.command == "inspect":
        payload = _broker_request(
            args.broker_url,
            run_id=args.run_id,
            command="inspect",
            payload={
                "observation_ids": args.observation_id,
                "artifact_roles": args.artifact_role,
                "max_bytes": args.max_bytes,
            },
        )
        payload = _materialize_broker_inspect(payload)
    elif args.command in {"pin", "unpin"}:
        payload = _broker_request(
            args.broker_url,
            run_id=args.run_id,
            command=args.command,
            payload={"observation_id": args.observation_id},
        )
    else:
        raise ValueError(
            f"{args.command} is an administrative operation and is not exposed "
            "through the launcher-owned memory broker"
        )
    _emit(payload)


def _dispatch(args: argparse.Namespace) -> None:
    if args.broker_url:
        if args.memory_root is not None:
            raise ValueError("--broker-url and --memory-root are mutually exclusive")
        _dispatch_broker(args)
        return
    memory = AgentMemory(run_id=args.run_id, memory_root=args.memory_root)

    if args.command == "init":
        _emit({"run_id": memory.run_id, "run_dir": str(memory.run_dir)})
    elif args.command == "record":
        _emit(memory.remember(_read_request(args.input)))
    elif args.command == "context":
        _emit(memory.context(limit=args.limit))
    elif args.command == "search":
        query = MemorySearchQuery(
            text=args.text,
            workflow=args.workflow,
            phase=args.phase,
            scene_revision_id=args.scene_revision_id,
            operation=args.operation,
            outcome=args.outcome,
            importance=args.importance,
            tag=args.tag,
            target=args.target,
            artifact_role=args.artifact_role,
            sequence_min=args.sequence_min,
            sequence_max=args.sequence_max,
            limit=args.limit,
        )
        _emit(memory.search(query))
    elif args.command == "inspect":
        _emit(
            memory.inspect(
                args.observation_id,
                artifact_roles=args.artifact_role,
                max_bytes=args.max_bytes,
            )
        )
    elif args.command == "pin":
        _emit(memory.pin(args.observation_id))
    elif args.command == "unpin":
        _emit(memory.unpin(args.observation_id))
    elif args.command == "rebuild":
        memory.rebuild_index()
        _emit({"run_id": memory.run_id, "rebuilt": True})
    elif args.command == "recover-torn-final-line":
        recovered = memory.recover_torn_final_line()
        _emit(
            {
                "run_id": memory.run_id,
                "recovered": recovered is not None,
                "preserved_path": str(recovered) if recovered else None,
            }
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _dispatch(args)
    except (MemoryError, KeyError, ValidationError, ValueError) as exc:
        _emit(
            {
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                },
            }
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
