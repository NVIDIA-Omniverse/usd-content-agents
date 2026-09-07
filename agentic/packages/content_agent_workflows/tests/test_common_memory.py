# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from content_agent_workflows.common import memory as memory_module
from content_agent_workflows.common import memory_cli
from content_agent_workflows.common.memory import (
    MAX_INSPECT_BYTES,
    AgentMemory,
    JournalCorruptionError,
    MemoryArtifactError,
    MemoryArtifactInput,
    MemoryBoundsError,
    MemoryError,
    MemoryExpectation,
    MemoryInteraction,
    MemoryOutcome,
    MemorySceneIdentity,
    MemorySearchQuery,
    RememberRequest,
    resolve_memory_root,
)


def _write_attempt(
    directory: Path,
    *,
    name: str,
    roughness: float,
    opacity: float,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    recipe = directory / f"{name}.json"
    recipe.write_text(
        json.dumps(
            {
                "material_name": "AmberGlass",
                "profile": "OpenPBR",
                "base_color": [0.95, 0.45, 0.08],
                "roughness": roughness,
                "opacity": opacity,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    preview = directory / f"{name}.png"
    alpha = round(opacity * 255)
    Image.new("RGBA", (8, 8), (242, 115, 20, alpha)).save(preview)
    return recipe, preview


def _remember_attempt(
    memory: AgentMemory,
    *,
    recipe: Path,
    preview: Path,
    classification: str,
    summary: str,
    parent_observation_id: str | None = None,
) -> str:
    record = memory.remember(
        RememberRequest(
            workflow="material-generation",
            phase="material-refinement",
            parent_observation_id=parent_observation_id,
            scene=MemorySceneIdentity(scene_revision_id="shader-ball-v1"),
            interaction=MemoryInteraction(
                operation="render_material_preview",
                target_prim_paths=("/Root/Sphere",),
            ),
            expectation=MemoryExpectation(
                expected_changes=("Amber glass should read as translucent.",),
                forbidden_changes=("Do not change the shader-ball fixture.",),
            ),
            artifacts=(
                MemoryArtifactInput(
                    path=recipe.resolve(),
                    role="material.recipe",
                    media_type="application/json",
                    provenance="native",
                ),
                MemoryArtifactInput(
                    path=preview.resolve(),
                    role="preview.rgb",
                    media_type="image/png",
                    provenance="native",
                ),
            ),
            outcome=MemoryOutcome(
                classification=classification,
                summary=summary,
                confidence=0.9,
            ),
            importance="high",
            tags=("material-generation", "amber-glass"),
        )
    )
    return record.observation_id


def test_material_generation_refines_from_durable_observation_memory(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "runs"
    evidence = tmp_path / "attempts"
    memory = AgentMemory(run_id="material-demo", memory_root=memory_root)

    recipe_one, preview_one = _write_attempt(
        evidence, name="attempt-01", roughness=0.45, opacity=1.0
    )
    rejected_id = _remember_attempt(
        memory,
        recipe=recipe_one,
        preview=preview_one,
        classification="contradicted",
        summary="Preview remains opaque amber instead of translucent glass.",
    )

    context = memory.context()
    assert context.unresolved_observation_ids == (rejected_id,)
    assert (
        memory.search(MemorySearchQuery(text="opaque amber"))[0].observation_id
        == rejected_id
    )
    assert (
        memory.search(MemorySearchQuery(scene_revision_id="shader-ball-v1"))[
            0
        ].observation_id
        == rejected_id
    )

    inspected = memory.inspect((rejected_id,), artifact_roles=("preview.rgb",))
    assert inspected.artifacts[0].path.read_bytes() == preview_one.read_bytes()
    assert inspected.artifacts[0].path.is_relative_to(memory_root / ".cache")

    recipe_two, preview_two = _write_attempt(
        evidence, name="attempt-02", roughness=0.12, opacity=0.32
    )
    accepted_id = _remember_attempt(
        memory,
        recipe=recipe_two,
        preview=preview_two,
        classification="matched",
        summary="Preview now reads as translucent amber glass on the same fixture.",
        parent_observation_id=rejected_id,
    )
    memory.pin(accepted_id)

    restarted = AgentMemory(run_id="material-demo", memory_root=memory_root)
    restarted_context = restarted.context(limit=2)
    assert [card.observation_id for card in restarted_context.cards] == [
        accepted_id,
        rejected_id,
    ]
    assert restarted_context.pinned_observation_ids == (accepted_id,)
    assert restarted_context.unresolved_observation_ids == (rejected_id,)

    journal = [
        json.loads(line)
        for line in restarted.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["sequence"] for event in journal] == [1, 2, 3]
    assert [event["kind"] for event in journal] == ["observation", "observation", "pin"]


def test_objects_deduplicate_and_index_rebuilds_from_journal(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="dedup", memory_root=tmp_path / "runs")
    recipe, preview = _write_attempt(
        tmp_path / "attempts", name="attempt", roughness=0.2, opacity=0.5
    )
    first_id = _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="ambiguous",
        summary="Preview needs another review pass.",
    )
    _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="not_checked",
        summary="Same evidence was carried into a validation pass.",
        parent_observation_id=first_id,
    )

    objects = [path for path in memory.objects_root.rglob("*") if path.is_file()]
    assert len(objects) == 2

    for suffix in ("", "-wal", "-shm"):
        Path(f"{memory.index_path}{suffix}").unlink(missing_ok=True)
    memory.rebuild_index()

    results = memory.search(MemorySearchQuery(tag="amber-glass"))
    assert len(results) == 2
    assert results[-1].observation_id == first_id


def test_parentless_remember_rebuilds_missing_index_before_projection(
    tmp_path: Path,
) -> None:
    memory = AgentMemory(run_id="append-recovery", memory_root=tmp_path / "runs")
    recipe, preview = _write_attempt(
        tmp_path / "attempts", name="attempt", roughness=0.2, opacity=0.5
    )
    first_id = _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="ambiguous",
        summary="First attempt needs review.",
    )
    second_id = _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="not_checked",
        summary="Second independent attempt awaits review.",
    )

    for suffix in ("", "-wal", "-shm"):
        Path(f"{memory.index_path}{suffix}").unlink(missing_ok=True)

    third_id = _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="matched",
        summary="Third independent attempt passed review.",
    )

    with sqlite3.connect(
        f"{memory.index_path.as_uri()}?mode=ro",
        uri=True,
    ) as connection:
        indexed_ids = [
            row[0]
            for row in connection.execute(
                "SELECT observation_id FROM observations ORDER BY sequence DESC"
            )
        ]
    assert indexed_ids == [third_id, second_id, first_id]

    results = memory.search(MemorySearchQuery(tag="amber-glass"))
    assert [item.observation_id for item in results] == [
        third_id,
        second_id,
        first_id,
    ]


def test_duplicate_artifact_rows_do_not_break_projection_or_rebuild(
    tmp_path: Path,
) -> None:
    memory = AgentMemory(run_id="duplicate-artifacts", memory_root=tmp_path / "runs")
    artifact = tmp_path / "recipe.json"
    artifact.write_text("{}", encoding="utf-8")
    duplicate = MemoryArtifactInput(
        path=artifact,
        role="material.recipe",
        media_type="application/json",
    )
    duplicate_with_different_metadata = MemoryArtifactInput(
        path=artifact,
        role="material.recipe",
        media_type="application/json",
        retention="pinned",
    )

    record = memory.remember(
        RememberRequest(
            workflow="material-generation",
            phase="generation",
            interaction=MemoryInteraction(operation="create_material_package"),
            artifacts=(duplicate, duplicate_with_different_metadata),
            outcome=MemoryOutcome(
                classification="not_checked",
                summary="Package awaits validation.",
            ),
        )
    )

    assert len(record.artifacts) == 1
    assert record.artifacts[0].retention == "pinned"
    assert memory.search(MemorySearchQuery(artifact_role="material.recipe"))
    memory.rebuild_index()
    assert memory.inspect((record.observation_id,)).observations == (record,)


def test_memory_rejects_symlink_artifacts(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="safe-artifact", memory_root=tmp_path / "runs")
    actual = tmp_path / "recipe.json"
    actual.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(actual)

    with pytest.raises(MemoryArtifactError, match="non-symlink"):
        memory.remember(
            RememberRequest(
                workflow="material-generation",
                phase="generation",
                interaction=MemoryInteraction(operation="create_material_package"),
                artifacts=(
                    MemoryArtifactInput(
                        path=linked.resolve(strict=False).parent / linked.name,
                        role="material.recipe",
                        media_type="application/json",
                    ),
                ),
                outcome=MemoryOutcome(
                    classification="not_checked",
                    summary="Package has not been previewed.",
                ),
            )
        )


def test_torn_final_line_requires_explicit_recovery(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="recovery", memory_root=tmp_path / "runs")
    recipe, preview = _write_attempt(
        tmp_path / "attempts", name="attempt", roughness=0.2, opacity=0.5
    )
    _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="not_checked",
        summary="Initial preview has not been assessed.",
    )
    with memory.journal_path.open("ab") as stream:
        stream.write(b'{"incomplete":')

    with pytest.raises(JournalCorruptionError, match="torn final line"):
        memory.context()

    restarted = AgentMemory(run_id="recovery", memory_root=tmp_path / "runs")
    backup = restarted.recover_torn_final_line()
    assert backup is not None
    assert backup.read_bytes() == b'{"incomplete":'
    assert restarted.context().cards


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_torn_line_recovery_refuses_linked_journals(
    tmp_path: Path,
    link_kind: str,
) -> None:
    memory = AgentMemory(run_id="linked-recovery", memory_root=tmp_path / "runs")
    outside = tmp_path / "outside.jsonl"
    original = b'{"outside":"must-not-be-truncated"}'
    outside.write_bytes(original)
    if link_kind == "symlink":
        memory.journal_path.symlink_to(outside)
    else:
        memory.journal_path.hardlink_to(outside)

    with pytest.raises(JournalCorruptionError, match="journal"):
        memory.recover_torn_final_line()

    assert outside.read_bytes() == original


def test_first_journal_append_fsyncs_its_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = AgentMemory(
        run_id="journal-directory-fsync", memory_root=tmp_path / "runs"
    )
    synced: list[Path] = []
    monkeypatch.setattr(memory_module, "_fsync_directory", synced.append)

    memory.remember(
        RememberRequest(
            workflow="material-generation",
            phase="generation",
            interaction=MemoryInteraction(operation="create_material_package"),
            outcome=MemoryOutcome(
                classification="not_checked",
                summary="The new journal entry must be durable.",
            ),
        )
    )

    assert synced == [memory.journal_path.parent]


def test_default_memory_root_uses_private_dot_memory_directory(tmp_path: Path) -> None:
    root = resolve_memory_root(repo_root=tmp_path, create=False)

    assert root == (tmp_path / "runs" / ".memory").resolve()
    assert not root.exists()


def test_default_memory_root_warns_about_unmigrated_pre_v06_memory(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "agentic" / "runs" / ".memory"
    legacy_root.mkdir(parents=True)

    with pytest.warns(RuntimeWarning, match="WU_AGENT_MEMORY_ROOT"):
        root = resolve_memory_root(repo_root=tmp_path, create=False)

    assert root == (tmp_path / "runs" / ".memory").resolve()
    assert not root.exists()


def test_inspect_rejects_bytes_above_hard_ceiling(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="bounded-inspect", memory_root=tmp_path / "runs")
    record = memory.remember(
        RememberRequest(
            workflow="material-generation",
            phase="generation",
            interaction=MemoryInteraction(operation="create_material_package"),
            outcome=MemoryOutcome(
                classification="not_checked",
                summary="The package awaits review.",
            ),
        )
    )

    with pytest.raises(MemoryBoundsError, match="cannot exceed"):
        memory.inspect(
            (record.observation_id,),
            max_bytes=MAX_INSPECT_BYTES + 1,
        )


def test_inspect_opportunistically_removes_expired_leases(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="lease-cleanup", memory_root=tmp_path / "runs")
    recipe, preview = _write_attempt(
        tmp_path / "attempts", name="attempt", roughness=0.2, opacity=0.5
    )
    record_id = _remember_attempt(
        memory,
        recipe=recipe,
        preview=preview,
        classification="not_checked",
        summary="Preview awaits review.",
    )
    inspected = memory.inspect(
        (record_id,),
        artifact_roles=("preview.rgb",),
    )
    assert inspected.lease_id is not None
    lease_dir = memory.cache_root / inspected.lease_id
    manifest_path = lease_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["expires_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    memory.inspect((record_id,))

    assert not lease_dir.exists()


def test_memory_refuses_symlinked_sqlite_index(tmp_path: Path) -> None:
    memory = AgentMemory(run_id="safe-index", memory_root=tmp_path / "runs")
    outside = tmp_path / "outside.sqlite"
    outside.write_bytes(b"do not touch")
    memory.index_path.symlink_to(outside)

    with pytest.raises(MemoryError, match="singly linked regular file"):
        memory.search(MemorySearchQuery(workflow="material-generation"))

    assert outside.read_bytes() == b"do not touch"


def test_memory_cli_emits_bounded_json_for_expected_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    returncode = memory_cli.main(
        [
            "--run-id",
            "cli-errors",
            "--memory-root",
            str(tmp_path / "runs"),
            "pin",
            "--observation-id",
            "obs-missing",
        ]
    )

    assert returncode == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "KeyError"
    assert "obs-missing" in payload["error"]["message"]


def test_memory_cli_emits_bounded_json_for_invalid_run_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    returncode = memory_cli.main(
        [
            "--run-id",
            "invalid/run-id",
            "--memory-root",
            str(tmp_path / "runs"),
            "init",
        ]
    )

    assert returncode == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "error"
    assert payload["error"]["type"] == "ValueError"
    assert "run_id must contain only" in payload["error"]["message"]


def test_legacy_trace_events_coexist_and_duplicate_memory_ids_fail_closed(
    tmp_path: Path,
) -> None:
    memory = AgentMemory(run_id="mixed-journal", memory_root=tmp_path / "runs")
    legacy = {
        "schema_version": "content-agents.trace.v1",
        "event_type": "run_created",
        "phase": "prepare",
        "summary": "Legacy trace event.",
    }
    memory.journal_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    memory.remember(
        RememberRequest(
            workflow="material-generation",
            phase="generation",
            interaction=MemoryInteraction(operation="create_material_package"),
            outcome=MemoryOutcome(
                classification="not_checked",
                summary="Package awaits preview.",
            ),
        )
    )

    lines = memory.journal_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["sequence"] == 1

    duplicate = json.loads(lines[1])
    duplicate["sequence"] = 2
    with memory.journal_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(duplicate) + "\n")

    with pytest.raises(JournalCorruptionError, match="repeats event_id"):
        memory.context()
