# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trust-boundary tests for the launcher-owned observation-memory broker."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import requests
from content_agent_workflows.common import memory_cli
from content_agent_workflows.common.memory import AgentMemory, MemorySearchQuery

from content_workflow_cli import memory_broker as memory_broker_module
from content_workflow_cli.memory_broker import (
    MEMORY_ORIGIN_AGENT_TAG,
    MEMORY_ORIGIN_LAUNCHER_TAG,
    AgentMemoryBroker,
)


def _record_payload(artifact: Path) -> dict[str, object]:
    return {
        "workflow": "mesh-segmentation",
        "phase": "revision_review",
        "interaction": {
            "operation": "review_mesh_segment_revision",
            "target_object_ids": ["fan blades"],
        },
        "outcome": {
            "classification": "ambiguous",
            "summary": "The blade tips still need a held-out view.",
        },
        "artifacts": [
            {
                "path": str(artifact),
                "role": "selected_only_review",
                "media_type": "image/png",
            }
        ],
    }


def test_cli_records_and_inspects_through_parent_owned_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    evidence = run_dir / "review.png"
    evidence.write_bytes(b"not-a-real-png-but-stable-evidence")
    request_path = run_dir / "observation.json"
    request_payload = _record_payload(evidence)
    request_payload["tags"] = [
        "mesh-segmentation",
        MEMORY_ORIGIN_LAUNCHER_TAG,
    ]
    request_path.write_text(json.dumps(request_payload), encoding="utf-8")
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(
        run_dir=run_dir,
        memory=memory,
        private_dir=tmp_path / "broker-private",
    )
    broker.start()
    monkeypatch.chdir(run_dir)
    try:
        assert (
            memory_cli.main(
                [
                    "--run-id",
                    "mesh-run",
                    "--broker-url",
                    broker.url,
                    "record",
                    "--input",
                    str(request_path),
                ]
            )
            == 0
        )
        recorded = json.loads(capsys.readouterr().out)
        observation_id = recorded["observation_id"]
        assert len(memory.search(MemorySearchQuery(target="fan blades"))) == 1
        assert len(memory.search(MemorySearchQuery(tag=MEMORY_ORIGIN_AGENT_TAG))) == 1
        assert memory.count_observations(tag=MEMORY_ORIGIN_AGENT_TAG) == 1
        assert memory.count_observations(tag=MEMORY_ORIGIN_LAUNCHER_TAG) == 0
        assert not (run_dir / ".objects").exists()
        assert not (run_dir / ".locks").exists()

        assert (
            memory_cli.main(
                [
                    "--run-id",
                    "mesh-run",
                    "--broker-url",
                    broker.url,
                    "inspect",
                    "--observation-id",
                    observation_id,
                    "--artifact-role",
                    "selected_only_review",
                ]
            )
            == 0
        )
        inspected = json.loads(capsys.readouterr().out)
        materialized = Path(inspected["artifacts"][0]["path"])
        assert materialized.is_relative_to(run_dir / ".agent-memory-inspect")
        assert materialized.read_bytes() == evidence.read_bytes()
        assert not any(memory.cache_root.iterdir())
    finally:
        broker.close()


def test_broker_rejects_symlinked_artifact_traversal(tmp_path: Path) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.png"
    secret.write_bytes(b"outside")
    (run_dir / "linked").symlink_to(outside, target_is_directory=True)
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(run_dir=run_dir, memory=memory)
    broker.start()
    try:
        response = requests.post(
            f"{broker.url}/v1/record",
            json={
                "run_id": "mesh-run",
                "request": _record_payload(run_dir / "linked" / "secret.png"),
            },
            timeout=10,
        )
        assert response.status_code == 400
        assert memory.count_observations() == 0
    finally:
        broker.close()


def test_broker_rejects_store_beneath_child_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    memory = AgentMemory(run_id="mesh-run", memory_root=run_dir / ".memory")

    with pytest.raises(ValueError, match="outside the child-writable"):
        AgentMemoryBroker(run_dir=run_dir, memory=memory)


def test_broker_rejects_content_change_when_metadata_appears_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    artifact = run_dir / "review.bin"
    artifact.write_bytes(b"before")
    fixed_metadata = artifact.stat()
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(run_dir=run_dir, memory=memory)
    original_read = memory_broker_module.os.read
    mutated = False

    def mutate_after_first_pass(file_descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(file_descriptor, size)
        if not chunk and not mutated:
            artifact.write_bytes(b"after!")
            mutated = True
        return chunk

    monkeypatch.setattr(memory_broker_module.os, "read", mutate_after_first_pass)
    monkeypatch.setattr(
        memory_broker_module.os,
        "fstat",
        lambda file_descriptor: fixed_metadata,
    )
    try:
        with pytest.raises(
            memory_broker_module.MemoryBrokerError,
            match="changed while reading",
        ):
            broker._snapshot_run_artifact(artifact)
    finally:
        broker.close()


def test_broker_is_healthy_and_blocks_search_until_plan_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    plan_path = run_dir / "part_plan.json"
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(
        run_dir=run_dir,
        memory=memory,
        search_ready_path=plan_path,
    )
    broker.start()
    try:
        health = requests.get(f"{broker.url}/health", timeout=10)
        assert health.status_code == 200
        assert health.headers["X-Content-Type-Options"] == "nosniff"
        assert health.json() == {"run_id": "mesh-run", "status": "ok"}

        payload = {
            "run_id": "mesh-run",
            "query": {"workflow": "mesh-segmentation", "target": "fan blades"},
        }
        blocked = requests.post(
            f"{broker.url}/v1/search",
            json=payload,
            timeout=10,
        )
        assert blocked.status_code == 409
        assert "until the initial part plan exists" in blocked.json()["error"]

        plan_path.write_text("{}", encoding="utf-8")
        allowed = requests.post(
            f"{broker.url}/v1/search",
            json=payload,
            timeout=10,
        )
        assert allowed.status_code == 200
        assert allowed.json() == []
        assert broker.operation_counts == {"search": 2}
    finally:
        broker.close()


def test_broker_maps_memory_bounds_to_bounded_client_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(run_dir=run_dir, memory=memory)
    broker.start()
    try:
        response = requests.post(
            f"{broker.url}/v1/context",
            json={"run_id": "mesh-run", "limit": 51},
            timeout=10,
        )
        assert response.status_code == 400
        assert len(response.json()["error"]) <= 1000
    finally:
        broker.close()


def test_broker_folds_unknown_commands_into_one_counter(tmp_path: Path) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")
    broker = AgentMemoryBroker(run_dir=run_dir, memory=memory)
    try:
        for index in range(10):
            with pytest.raises(memory_broker_module.MemoryBrokerError, match="unknown"):
                broker.dispatch(f"unknown-{index}", {"run_id": "mesh-run"})
        assert broker.operation_counts == {"unknown": 10}
    finally:
        broker.close()


def test_broker_tightens_existing_private_directory_permissions(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "child-run"
    run_dir.mkdir()
    private_dir = tmp_path / "broker-private"
    private_dir.mkdir(mode=0o755)
    private_dir.chmod(0o755)
    memory = AgentMemory(run_id="mesh-run", memory_root=tmp_path / "memory")

    broker = AgentMemoryBroker(
        run_dir=run_dir,
        memory=memory,
        private_dir=private_dir,
    )
    try:
        assert stat.S_IMODE(private_dir.stat().st_mode) == 0o700
    finally:
        broker.close()
