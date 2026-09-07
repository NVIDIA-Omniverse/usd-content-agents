# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Promoted tuned USDs must not depend on the doomed broker workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from content_workflow_cli import runner


def test_promotion_rejects_dependencies_inside_broker_workspace(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "sweep_work"
    candidates = work_root / "candidates"
    candidates.mkdir(parents=True)
    texture = candidates / "physics_assets" / "albedo.png"
    texture.parent.mkdir(parents=True)
    texture.write_bytes(b"png-bytes")

    target_dir = tmp_path / "run"
    target_dir.mkdir()
    promoted = target_dir / "physics.usda"
    promoted.write_text(
        f'#usda 1.0\ndef Material "M" {{\n    asset inputs:file = @{texture}@\n}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="broker workspace"):
        runner._assert_promoted_dependencies_survive_broker_close(
            promoted, broker_work_root=work_root
        )


def test_promotion_accepts_localized_dependency_closure(tmp_path: Path) -> None:
    work_root = tmp_path / "sweep_work"
    work_root.mkdir()
    target_dir = tmp_path / "run"
    sidecar = target_dir / "physics_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"png-bytes")
    promoted = target_dir / "physics.usda"
    promoted.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./physics_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )
    runner._assert_promoted_dependencies_survive_broker_close(
        promoted, broker_work_root=work_root
    )


def _tuning_config(tmp_path: Path, run_dir: Path, output_usd_path: Path):
    from content_workflow_cli.runner import PhysicsApplyConfig

    source = tmp_path / "source.usda"
    if not source.exists():
        source.write_text("#usda 1.0\n", encoding="utf-8")
    return PhysicsApplyConfig(
        repo_root=tmp_path,
        usd_path=source,
        output_dir=run_dir,
        output_usd_path=output_usd_path,
    )


def _mark_owned(sidecar_dir: Path) -> Path:
    """Write the portable-sidecar ownership marker the promotion gate requires."""

    from world_understanding.functions.graphics.so_export import (
        PORTABLE_SIDECAR_MARKER_BYTES,
        PORTABLE_SIDECAR_MARKER_NAME,
    )

    sidecar_dir.mkdir(parents=True, exist_ok=True)
    marker = sidecar_dir / PORTABLE_SIDECAR_MARKER_NAME
    marker.write_bytes(PORTABLE_SIDECAR_MARKER_BYTES)
    return marker


def test_promotion_replaces_stale_target_sidecar_members(tmp_path: Path) -> None:
    """Revalidation ran against the candidate bundle; a same-named stale file
    already at the target must be digest-verified and REPLACED (with a
    pretune backup), never silently retained."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"fresh-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    # Stale bytes at the CANONICAL sidecar name the promoted member lands on.
    stale = out_dir / "physics.usda_assets" / "albedo.png"
    stale.parent.mkdir(parents=True)
    _mark_owned(stale.parent)
    stale.write_bytes(b"stale-bytes")
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    promotion.commit()
    assert promotion.target == target
    assert stale.read_bytes() == b"fresh-bytes"
    backups = [
        path.read_bytes() for path in (run_dir / "tuning").glob("pretune_*albedo*")
    ]
    assert b"stale-bytes" in backups


def test_promotion_failure_leaves_the_canonical_root_untouched(
    tmp_path: Path,
) -> None:
    """A candidate whose dependency escapes the candidate directory fails
    BEFORE the canonical root is replaced - the previous output survives."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    candidates.mkdir(parents=True)
    escaped = work / "escaped.png"
    escaped.write_bytes(b"outside-candidate-dir")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        f'#usda 1.0\ndef Material "M" {{\n    asset inputs:file = @{escaped}@\n}}\n',
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usda"
    target.write_text("#usda 1.0\n# previous\n", encoding="utf-8")

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises(RuntimeError, match="escapes the candidate"):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=sha256_file(tuned),
        )
    assert target.read_text(encoding="utf-8") == "#usda 1.0\n# previous\n"


def test_promotion_renames_sidecar_to_the_canonical_output(
    tmp_path: Path,
) -> None:
    """Downstream bundling discovers "<output-name>_assets"; the promoted
    root must reference that name, not the broker's trial_XXXX sidecar."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"fresh-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    promotion.commit()
    assert promotion.target == target
    text = target.read_text(encoding="utf-8")
    assert "trial_0001" not in text
    # Sdf normalizes the "./" prefix away when saving.
    assert "physics.usda_assets/albedo.png" in text
    assert (
        out_dir / "physics.usda_assets" / "albedo.png"
    ).read_bytes() == b"fresh-bytes"


def test_promotion_removes_stale_extra_sidecar_members(tmp_path: Path) -> None:
    """The canonical sidecar is workflow-owned and ships wholesale via the
    downstream bundle writer: after promotion it must contain EXACTLY the
    new closure, so extra files already in an authored sidecar are removed
    (with a pretune backup), never silently retained."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"fresh-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    target_sidecar = out_dir / "physics.usda_assets"
    _mark_owned(target_sidecar)
    # An authored sidecar with a member the new closure does not reference.
    stale_extra = target_sidecar / "textures" / "leftover.png"
    stale_extra.parent.mkdir(parents=True)
    stale_extra.write_bytes(b"stale-extra-bytes")
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    promotion.commit()
    assert promotion.target == target
    # The sidecar holds EXACTLY the new closure - no leftovers.
    members = sorted(
        path.relative_to(target_sidecar).as_posix()
        for path in target_sidecar.rglob("*")
        if path.is_file()
    )
    # The ownership marker survives pruning; USD members are exactly the
    # new closure.
    assert members == [".usd_portable_sidecar", "albedo.png"]
    # The removed member was backed up under tuning/pretune_*.
    backups = [
        path.read_bytes() for path in (run_dir / "tuning").glob("pretune_*leftover*")
    ]
    assert b"stale-extra-bytes" in backups


def test_promotion_rolls_back_staged_members_on_failure(tmp_path: Path) -> None:
    """A failure after some closure members were replaced must restore the
    previous bytes - the old canonical root may still reference them."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    kept = sidecar / "albedo.png"
    kept.write_bytes(b"fresh-bytes")
    doomed = sidecar / "zz_late.png"
    doomed.write_bytes(b"late-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"
        "    asset inputs:late = @./trial_0001.usda_assets/zz_late.png@\n"
        "}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    stale = out_dir / "physics.usda_assets" / "albedo.png"
    stale.parent.mkdir(parents=True)
    _mark_owned(stale.parent)
    stale.write_bytes(b"stale-bytes")
    target = out_dir / "physics.usda"
    target.write_text("#usda 1.0\n# previous\n", encoding="utf-8")

    from content_workflow_cli.tuning_broker import sha256_file

    digest = sha256_file(tuned)
    # Delete one dependency source AFTER digest capture: staging replaces the
    # stale member first, then fails on the missing one.
    doomed.unlink()

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises((RuntimeError, OSError)):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=digest,
        )
    # The replaced member was rolled back and the canonical root survived.
    assert stale.read_bytes() == b"stale-bytes"
    assert target.read_text(encoding="utf-8") == "#usda 1.0\n# previous\n"


def test_promotion_rejects_symlinked_canonical_sidecar(tmp_path: Path) -> None:
    """The run directory stays child-writable during tuning: a symlink
    planted at the canonical sidecar must refuse promotion outright, never
    redirect staging/cleanup writes outside the output tree."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"fresh-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )

    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim = victim_dir / "albedo.png"
    victim.write_bytes(b"victim-bytes")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "physics.usda_assets").symlink_to(victim_dir)
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises(RuntimeError, match="symlinked canonical sidecar"):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=sha256_file(tuned),
        )
    # Nothing outside the output tree was touched and no root was written.
    assert victim.read_bytes() == b"victim-bytes"
    assert not target.exists()


def test_promotion_rejects_symlinked_nested_sidecar_directory(
    tmp_path: Path,
) -> None:
    """A symlink NESTED inside a real canonical sidecar is equally
    tampering: no-follow validation must reject it before any mutation."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True)
    (sidecar / "albedo.png").write_bytes(b"fresh-bytes")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"
        "}\n",
        encoding="utf-8",
    )

    victim_dir = tmp_path / "victim"
    victim_dir.mkdir()
    victim = victim_dir / "stale.png"
    victim.write_bytes(b"victim-bytes")
    out_dir = tmp_path / "out"
    target_sidecar = out_dir / "physics.usda_assets"
    target_sidecar.mkdir(parents=True)
    (target_sidecar / "textures").symlink_to(victim_dir)
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=sha256_file(tuned),
        )
    # The symlink target survived: stale-cleanup never followed the link.
    assert victim.read_bytes() == b"victim-bytes"
    assert (target_sidecar / "textures").is_symlink()
    assert not target.exists()


def test_promotion_blocks_when_candidate_inspection_fails(tmp_path: Path) -> None:
    """Dependency inspection errors must FAIL promotion: replacing the root
    without a stageable closure lets broker.close() delete the workspace the
    promoted asset still resolves into."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usdc"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    candidates.mkdir(parents=True)
    # Text bytes in a crate container: pxr cannot inspect this closure.
    tuned = candidates / "trial_0001.usdc"
    tuned.write_text("#usda 1.0\n# not a crate file\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usdc"
    target.write_text("#usda 1.0\n# previous\n", encoding="utf-8")

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises(RuntimeError, match="closure cannot be staged"):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=sha256_file(tuned),
        )
    assert target.read_text(encoding="utf-8") == "#usda 1.0\n# previous\n"


def test_final_closure_gate_blocks_on_uninspectable_root(tmp_path: Path) -> None:
    """The staged-bundle gate fails CLOSED when inspection errors."""

    staged = tmp_path / "staged.usdc"
    staged.write_text("#usda 1.0\n# not a crate file\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="closure cannot be verified"):
        runner._assert_promoted_dependencies_survive_broker_close(
            staged, broker_work_root=tmp_path / "sweep_work"
        )


def _write_candidate(
    candidates: Path, *, albedo: bytes, late: bytes | None = None
) -> Path:
    sidecar = candidates / "trial_0001.usda_assets"
    sidecar.mkdir(parents=True, exist_ok=True)
    (sidecar / "albedo.png").write_bytes(albedo)
    refs = ["    asset inputs:file = @./trial_0001.usda_assets/albedo.png@\n"]
    if late is not None:
        (sidecar / "zz_late.png").write_bytes(late)
        refs.append("    asset inputs:late = @./trial_0001.usda_assets/zz_late.png@\n")
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        '#usda 1.0\ndef Material "M" {\n' + "".join(refs) + "}\n",
        encoding="utf-8",
    )
    return tuned


def test_repeated_promotion_rolls_back_to_current_attempt_bytes(
    tmp_path: Path,
) -> None:
    """The tuning/pretune_* archive keeps the FIRST bytes ever seen at a
    destination; a failed second promotion into the same run_dir must roll
    back to the bytes present at the start of the SECOND attempt, never to
    that stale archive."""

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    stale = out_dir / "physics.usda_assets" / "albedo.png"
    stale.parent.mkdir(parents=True)
    _mark_owned(stale.parent)
    stale.write_bytes(b"stale-bytes")
    target = out_dir / "physics.usda"
    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)

    first = _write_candidate(tmp_path / "work1" / "candidates", albedo=b"first-bytes")
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=first,
        usd_sha256=sha256_file(first),
    )
    promotion.commit()
    assert stale.read_bytes() == b"first-bytes"
    first_root_bytes = target.read_bytes()

    second = _write_candidate(
        tmp_path / "work2" / "candidates", albedo=b"second-bytes", late=b"late-bytes"
    )
    second_digest = sha256_file(second)
    # Delete one dependency source AFTER digest capture: staging replaces
    # albedo first, then fails on the missing member.
    (second.parent / "trial_0001.usda_assets" / "zz_late.png").unlink()
    with pytest.raises((RuntimeError, OSError)):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=second,
            usd_sha256=second_digest,
        )
    # Rolled back to the SECOND attempt's starting bytes, not the archive.
    assert stale.read_bytes() == b"first-bytes"
    assert target.read_bytes() == first_root_bytes
    # The one-time archive still preserves the original pre-tuning bytes.
    backups = [
        path.read_bytes() for path in (run_dir / "tuning").glob("pretune_*albedo*")
    ]
    assert backups == [b"stale-bytes"]


def _vomp_baseline_stub(run_dir: Path, tuned: Path):
    """Minimal attested-baseline double for the publication transaction."""

    from types import SimpleNamespace

    from content_agent_workflows.physics import PhysicsVompMassResult

    from content_workflow_cli.tuning_broker import sha256_file

    provenance = run_dir / "raw" / "physics_vomp_mass_properties.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_bytes(b"{}\n")
    result = PhysicsVompMassResult(
        target_prim_path="/World",
        input_usd_path=str(run_dir / "raw" / "pre_vomp.usda"),
        output_usd_path=str(tuned),
        output_usd_sha256=sha256_file(tuned),
        provenance_path=str(provenance),
        provenance_sha256=sha256_file(provenance),
        evidence_dir=str(run_dir / "vomp" / "evidence"),
        evidence_manifest_path=str(run_dir / "vomp" / "evidence" / "manifest.json"),
        vomp_npz_path=str(run_dir / "vomp" / "evidence" / "vomp.npz"),
        worker_manifest_path=str(run_dir / "vomp" / "evidence" / "worker.json"),
        worker_log_path=str(run_dir / "vomp" / "evidence" / "worker.log"),
        sample_count=8,
        mass_kg=1.0,
        center_of_mass_local_m=(0.0, 0.0, 0.0),
        diagonal_inertia_kg_m2=(1.0, 1.0, 1.0),
        principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    canonical = run_dir / "raw" / "physics_vomp_result.json"
    canonical.write_text(result.model_dump_json(), encoding="utf-8")
    return SimpleNamespace(
        result=result,
        canonical_result_path=canonical,
        provenance_bytes=b"{}\n",
    )


def test_publication_binds_vomp_result_to_final_promoted_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Promotion rewrites the staged root's sidecar references, so the
    published VoMP result must carry the digest of the FINAL on-disk root,
    not the materialized candidate's - attestation verifies against disk."""

    import content_agent_workflows.physics as physics_workflows
    from content_agent_workflows.physics import PhysicsVompMassResult

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    tuned = _write_candidate(
        tmp_path / "sweep_work" / "candidates", albedo=b"fresh-bytes"
    )
    tuned_sha256 = sha256_file(tuned)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usda"

    baseline = _vomp_baseline_stub(run_dir, tuned)
    monkeypatch.setattr(
        physics_workflows,
        "verify_vomp_mass_properties",
        lambda _usd, result, *, provenance_bytes=None: {"verified": True},
    )
    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    publication = runner._publish_promoted_physics_tuning(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        tuned_usd_sha256=tuned_sha256,
        vomp_baseline=baseline,
    )
    assert publication.error is None
    assert publication.promoted_path == target
    published = PhysicsVompMassResult.model_validate_json(
        baseline.canonical_result_path.read_bytes()
    )
    # The rewrite changed the root bytes; the published digest matches the
    # promoted root on disk, never the pre-rewrite candidate.
    assert published.output_usd_sha256 == sha256_file(target)
    assert published.output_usd_sha256 != tuned_sha256
    assert published.output_usd_path == str(target)


def test_publication_failure_restores_sidecar_and_canonical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VoMP publication failure AFTER promotion must restore the sidecar
    mutations together with the canonical files - never the old root paired
    with a replaced or deleted dependency closure."""

    import content_agent_workflows.physics as physics_workflows

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    finalized_bytes = finalized.read_bytes()

    tuned = _write_candidate(
        tmp_path / "sweep_work" / "candidates",
        albedo=b"fresh-bytes",
        late=b"late-bytes",
    )
    out_dir = tmp_path / "out"
    target = out_dir / "physics.usda"
    target_sidecar = out_dir / "physics.usda_assets"
    target_sidecar.mkdir(parents=True)
    _mark_owned(target_sidecar)
    (target_sidecar / "albedo.png").write_bytes(b"old-bytes")
    (target_sidecar / "leftover.png").write_bytes(b"stale-extra-bytes")
    target.write_text("#usda 1.0\n# previous\n", encoding="utf-8")

    baseline = _vomp_baseline_stub(run_dir, tuned)
    canonical_bytes = baseline.canonical_result_path.read_bytes()

    def failing_verify(_usd, result, *, provenance_bytes=None):
        raise RuntimeError("post-copy VoMP verification failed")

    monkeypatch.setattr(
        physics_workflows, "verify_vomp_mass_properties", failing_verify
    )
    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    publication = runner._publish_promoted_physics_tuning(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        tuned_usd_sha256=sha256_file(tuned),
        vomp_baseline=baseline,
    )
    assert publication.error is not None
    assert "rolled back" in publication.error
    assert publication.promoted_path is None
    # Canonical files restored.
    assert target.read_text(encoding="utf-8") == "#usda 1.0\n# previous\n"
    assert finalized.read_bytes() == finalized_bytes
    assert baseline.canonical_result_path.read_bytes() == canonical_bytes
    # The sidecar is restored EXACTLY: replaced members carry their previous
    # bytes, removed extras are back, and newly added members are gone.
    members = {
        path.name: path.read_bytes()
        for path in target_sidecar.rglob("*")
        if path.is_file() and not path.name.startswith(".usd_portable_sidecar")
    }
    assert members == {
        "albedo.png": b"old-bytes",
        "leftover.png": b"stale-extra-bytes",
    }


def test_promotion_refuses_unowned_target_sidecar(tmp_path: Path) -> None:
    """The conventional "<output-name>_assets" NAME is not ownership proof:
    a pre-existing directory without the portable-sidecar marker may be an
    unrelated sibling and must never be staged into or pruned."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    tuned = _write_candidate(tmp_path / "work" / "candidates", albedo=b"fresh-bytes")

    out_dir = tmp_path / "out"
    unowned = out_dir / "physics.usda_assets"
    unowned.mkdir(parents=True)
    user_data = unowned / "user_data.png"
    user_data.write_bytes(b"user-bytes")
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    with pytest.raises(RuntimeError, match="unowned sidecar"):
        runner._promote_tuned_physics_usd(
            config=config,
            run_dir=run_dir,
            finalized_physics_usd=finalized,
            tuned_usd=tuned,
            usd_sha256=sha256_file(tuned),
        )
    # Nothing in the unrelated directory was touched and no root was written.
    assert sorted(path.name for path in unowned.iterdir()) == ["user_data.png"]
    assert user_data.read_bytes() == b"user-bytes"
    assert not target.exists()


def test_promotion_prunes_owned_sidecar_and_keeps_marker(tmp_path: Path) -> None:
    """A marker-owned sidecar is pruned to exactly the new closure and the
    ownership marker survives (it is workflow metadata, not a USD dep);
    a NEW sidecar created by promotion gains the marker."""

    from world_understanding.functions.graphics.so_export import (
        PORTABLE_SIDECAR_MARKER_BYTES,
        PORTABLE_SIDECAR_MARKER_NAME,
    )

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    tuned = _write_candidate(tmp_path / "work" / "candidates", albedo=b"fresh-bytes")

    out_dir = tmp_path / "out"
    owned = out_dir / "physics.usda_assets"
    _mark_owned(owned)
    (owned / "leftover.png").write_bytes(b"stale-extra-bytes")
    target = out_dir / "physics.usda"

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    promotion.commit()
    members = sorted(path.name for path in owned.rglob("*") if path.is_file())
    assert members == [PORTABLE_SIDECAR_MARKER_NAME, "albedo.png"]
    marker = owned / PORTABLE_SIDECAR_MARKER_NAME
    assert marker.read_bytes() == PORTABLE_SIDECAR_MARKER_BYTES

    # A brand-new sidecar created by a second promotion elsewhere also gains
    # the ownership marker for future portable exports.
    fresh_out = tmp_path / "fresh_out"
    fresh_out.mkdir()
    fresh_target = fresh_out / "physics.usda"
    second = _write_candidate(tmp_path / "work2" / "candidates", albedo=b"v2-bytes")
    config2 = _tuning_config(tmp_path, run_dir, output_usd_path=fresh_target)
    runner._promote_tuned_physics_usd(
        config=config2,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=second,
        usd_sha256=sha256_file(second),
    ).commit()
    fresh_marker = fresh_out / "physics.usda_assets" / PORTABLE_SIDECAR_MARKER_NAME
    assert fresh_marker.read_bytes() == PORTABLE_SIDECAR_MARKER_BYTES


def test_sidecar_rollback_failure_raises_and_retains_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed member restore must RAISE (never report a clean rollback)
    and retain the per-attempt snapshot directory for recovery."""

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    tuned = _write_candidate(tmp_path / "work" / "candidates", albedo=b"fresh-bytes")

    out_dir = tmp_path / "out"
    stale = out_dir / "physics.usda_assets" / "albedo.png"
    stale.parent.mkdir(parents=True)
    _mark_owned(stale.parent)
    stale.write_bytes(b"stale-bytes")
    target = out_dir / "physics.usda"

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    assert any(backup is not None for _dest, backup in promotion.staged_changes)

    real_copy2 = runner.shutil.copy2

    def failing_copy2(source, destination, **kwargs):  # noqa: ANN001, ANN003
        raise OSError("injected restore failure")

    monkeypatch.setattr(runner.shutil, "copy2", failing_copy2)
    with pytest.raises(RuntimeError, match="could not restore"):
        promotion.rollback()
    monkeypatch.setattr(runner.shutil, "copy2", real_copy2)
    # The snapshots remain recoverable on disk.
    assert promotion.rollback_dir.is_dir()
    assert any(promotion.rollback_dir.iterdir())


def test_final_review_staging_includes_the_promoted_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged final-review copy must carry the FINAL promoted digest and
    the complete sidecar so its localized references resolve in place."""

    import content_agent_workflows.physics as physics_workflows
    from pxr import UsdUtils

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    tuned = _write_candidate(
        tmp_path / "sweep_work" / "candidates", albedo=b"fresh-bytes"
    )
    tuned_sha256 = sha256_file(tuned)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usda"

    baseline = _vomp_baseline_stub(run_dir, tuned)
    monkeypatch.setattr(
        physics_workflows,
        "verify_vomp_mass_properties",
        lambda _usd, result, *, provenance_bytes=None: {"verified": True},
    )
    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    publication = runner._publish_promoted_physics_tuning(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        tuned_usd_sha256=tuned_sha256,
        vomp_baseline=baseline,
    )
    assert publication.error is None
    assert publication.transaction is not None
    # The FINAL promoted-root digest (post-rewrite) is what review verifies.
    assert publication.promoted_sha256 == sha256_file(target)
    assert publication.promoted_sha256 != tuned_sha256

    # Staging with the pre-rewrite candidate digest would reject; the final
    # digest stages successfully WITH the sidecar closure.
    staged = runner._stage_promoted_physics_usd_for_final_review(
        run_dir=run_dir,
        promoted_usd=target,
        expected_sha256=publication.promoted_sha256,
    )
    assert staged.read_bytes() == target.read_bytes()
    staged_member = staged.parent / "physics.usda_assets" / "albedo.png"
    assert staged_member.read_bytes() == b"fresh-bytes"
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(staged))
    assert not unresolved
    staged_root = staged.parent.resolve()
    for identifier in [layer.identifier for layer in layers] + [
        str(asset) for asset in assets
    ]:
        Path(identifier).resolve().relative_to(staged_root)
    publication.transaction.commit()


def test_final_review_rejection_rolls_back_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promotion transaction survives publication: rolling it back after
    a (simulated) final-review rejection restores the sidecar exactly."""

    import content_agent_workflows.physics as physics_workflows

    from content_workflow_cli.tuning_broker import sha256_file

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")
    tuned = _write_candidate(
        tmp_path / "sweep_work" / "candidates", albedo=b"fresh-bytes"
    )
    out_dir = tmp_path / "out"
    target = out_dir / "physics.usda"
    target_sidecar = out_dir / "physics.usda_assets"
    _mark_owned(target_sidecar)
    (target_sidecar / "albedo.png").write_bytes(b"old-bytes")
    (target_sidecar / "leftover.png").write_bytes(b"stale-extra-bytes")
    target.write_text("#usda 1.0\n# previous\n", encoding="utf-8")

    baseline = _vomp_baseline_stub(run_dir, tuned)
    monkeypatch.setattr(
        physics_workflows,
        "verify_vomp_mass_properties",
        lambda _usd, result, *, provenance_bytes=None: {"verified": True},
    )
    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    publication = runner._publish_promoted_physics_tuning(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        tuned_usd_sha256=sha256_file(tuned),
        vomp_baseline=baseline,
    )
    assert publication.error is None
    assert publication.transaction is not None
    # Publication succeeded: sidecar carries the new closure.
    assert (target_sidecar / "albedo.png").read_bytes() == b"fresh-bytes"
    assert not (target_sidecar / "leftover.png").exists()

    # Simulated final-review rejection: the still-live transaction restores
    # the sidecar exactly (replaced members, removed extras).
    publication.transaction.rollback()
    members = {
        path.name: path.read_bytes()
        for path in target_sidecar.rglob("*")
        if path.is_file() and not path.name.startswith(".usd_portable_sidecar")
    }
    assert members == {
        "albedo.png": b"old-bytes",
        "leftover.png": b"stale-extra-bytes",
    }


def test_promotion_exempts_resolver_owned_assets(tmp_path: Path) -> None:
    """A candidate carrying bare MDL tokens and resolver URIs promotes: those
    references stay runtime-resolved and are neither staged nor treated as
    escaped/unresolved by the closure gates."""

    run_dir = tmp_path / "run"
    finalized = run_dir / "physics.usda"
    finalized.parent.mkdir(parents=True)
    finalized.write_text("#usda 1.0\n# finalized\n", encoding="utf-8")

    work = tmp_path / "sweep_work"
    candidates = work / "candidates"
    candidates.mkdir(parents=True)
    tuned = candidates / "trial_0001.usda"
    tuned.write_text(
        "#usda 1.0\n"
        'def Material "M" {\n'
        "    asset inputs:mdl = @OmniPBR.mdl@\n"
        "    asset inputs:remote = @omniverse://server/materials/base.usd@\n"
        "}\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    target = out_dir / "physics.usda"

    from content_workflow_cli.tuning_broker import sha256_file

    config = _tuning_config(tmp_path, run_dir, output_usd_path=target)
    promotion = runner._promote_tuned_physics_usd(
        config=config,
        run_dir=run_dir,
        finalized_physics_usd=finalized,
        tuned_usd=tuned,
        usd_sha256=sha256_file(tuned),
    )
    promoted = promotion.target if hasattr(promotion, "target") else promotion[0]
    assert Path(str(promoted)) == target
    text = target.read_text(encoding="utf-8")
    assert "OmniPBR.mdl" in text


def test_sidecar_marker_check_is_bounded(tmp_path: Path) -> None:
    """A child-planted huge sparse file at the marker path must be refused
    at fstat size-check without allocating the whole file."""

    from world_understanding.functions.graphics.so_export import (
        PORTABLE_SIDECAR_MARKER_BYTES,
        PORTABLE_SIDECAR_MARKER_NAME,
    )

    marker = tmp_path / PORTABLE_SIDECAR_MARKER_NAME
    marker.write_bytes(PORTABLE_SIDECAR_MARKER_BYTES)
    assert runner._portable_sidecar_marker_matches(marker) is True

    with open(marker, "wb") as handle:
        handle.truncate(64 * 1024 * 1024 * 1024)
    assert runner._portable_sidecar_marker_matches(marker) is False

    marker.unlink()
    import os as _os

    _os.mkfifo(marker)
    assert runner._portable_sidecar_marker_matches(marker) is False
