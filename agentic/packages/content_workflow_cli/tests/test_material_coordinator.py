# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coordinator-owned Material workflow policy and usd-cli wiring."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from content_agent_workflows.material_assignment import (
    write_material_grounding_diagnostics,
)
from PIL import Image
from pxr import Tf

from content_workflow_cli import material_coordinator, runner
from content_workflow_cli.material_coordinator import (
    finalize_material_for_coordinator as _finalize_material_for_coordinator,
)
from content_workflow_cli.material_coordinator import (
    prepare_material_for_coordinator,
)
from content_workflow_cli.material_coordinator import (
    release_material_for_coordinator as _release_material_for_coordinator,
)
from content_workflow_cli.material_coordinator import (
    review_material_for_coordinator as _review_material_for_coordinator,
)
from content_workflow_cli.runner import MaterialAssignConfig


def _config(tmp_path: Path) -> MaterialAssignConfig:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "cabinet.usda"
    source.write_text('#usda 1.0\ndef Xform "Cabinet" {}\n', encoding="utf-8")
    manifest = repo / "materials.yaml"
    manifest.write_text(
        "entries:\n  - name: paint\n    binding: /Materials/Paint\n",
        encoding="utf-8",
    )
    library = repo / "materials.usda"
    library.write_text(
        '#usda 1.0\n\ndef Scope "Materials"\n{\n    def Material "Paint"\n    {\n    }\n}\n',
        encoding="utf-8",
    )
    image = repo / "appearance.data"
    image.write_bytes(b"image reference")
    notes = repo / "notes.png"
    notes.write_text("text reference", encoding="utf-8")
    return MaterialAssignConfig(
        repo_root=repo,
        usd_path=source,
        reference_images=[image],
        reference_files=[notes],
        materials_yaml=manifest,
        materials_usd=library,
        output_dir=tmp_path / "run",
        output_usd_path=tmp_path / "materialized.usda",
        optimize=False,
        preflight=True,
    )


def _manifest_sha256(path: Path) -> str:
    return material_coordinator.file_sha256(path)


def finalize_material_for_coordinator(
    run_dir: Path,
    *,
    decision_patch_path: Path,
) -> material_coordinator.MaterialCoordinatorReviewRequired:
    return _finalize_material_for_coordinator(
        run_dir,
        decision_patch_path=decision_patch_path,
        preparation_sha256=_manifest_sha256(run_dir / "coordinator_preparation.json"),
    )


def review_material_for_coordinator(
    run_dir: Path,
    *,
    review_patch_path: Path,
) -> material_coordinator.MaterialCoordinatorResult:
    return _review_material_for_coordinator(
        run_dir,
        review_patch_path=review_patch_path,
        preparation_sha256=_manifest_sha256(run_dir / "coordinator_preparation.json"),
        application_receipt_sha256=_manifest_sha256(
            run_dir / "raw" / "material_application_receipt.json"
        ),
    )


def release_material_for_coordinator(
    run_dir: Path,
) -> material_coordinator.MaterialCoordinatorRelease:
    receipt_path = run_dir / "raw" / "material_application_receipt.json"
    return _release_material_for_coordinator(
        run_dir,
        preparation_sha256=_manifest_sha256(run_dir / "coordinator_preparation.json"),
        application_receipt_sha256=(
            _manifest_sha256(receipt_path) if receipt_path.is_file() else None
        ),
    )


class _UsdCliSession:
    """A session-shaped low-level adapter; it deliberately has no REST behavior."""

    session_id = "workflow-material-session"

    def __init__(self, run_dir: Path, *, renderer: str = "ovrtx") -> None:
        self.run_dir = run_dir
        self.renderer = renderer
        self.closed = 0
        self.opened: list[Path] = []
        self.open_force_reloads: list[bool] = []
        self.commands: list[list[str]] = []
        self.command_timeouts: list[float | None] = []
        self.render_timeouts: list[float] = []
        self.bound_materials: dict[str, str] = {}
        self.bound_material_paths: dict[str, str] = {}
        self.binding_overrides: dict[str, str | None] = {}
        self.checkpoint_sequence = 0
        self.material_audit_response: dict[str, object] = {
            "ok": True,
            "data": {
                "counts": {"invalid_binding_targets": 0},
                "renderables": [],
            },
        }

    @property
    def receipt_checkpoint_file(self) -> Path:
        checkpoint = self.run_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            self._advance_checkpoint("created")
        return checkpoint

    def _advance_checkpoint(self, operation: str) -> None:
        checkpoint = self.run_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint.is_file():
            existing = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.checkpoint_sequence = max(
                self.checkpoint_sequence,
                int(existing.get("sequence", 0)),
            )
        self.checkpoint_sequence += 1
        checkpoint.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "content-agent-workflows.usd-cli-receipt-checkpoint.v1"
                    ),
                    "session_id": self.session_id,
                    "sequence": self.checkpoint_sequence,
                    "operation": operation,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def open(
        self,
        scene: Path,
        *,
        read_only: bool = False,
        force_reload: bool = False,
    ) -> dict[str, object]:
        del read_only
        self._advance_checkpoint("open")
        self.opened.append(scene)
        self.open_force_reloads.append(force_reload)
        return {"ok": True}

    def require_ovrtx(self, output_dir: Path) -> dict[str, object]:
        self._advance_checkpoint("require_ovrtx")
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / "ovrtx_readiness_probe.png"
        Image.new("RGB", (64, 64), "green").save(image)
        result: dict[str, object] = {
            "schema_version": "usd-cli.render-probe.v1",
            "resolved_renderer": self.renderer,
            "engine": "ovrtx",
            "transport": "remote" if self.renderer == "remote" else "local",
            "ready": True,
            "render": {"path": str(image), "backend": self.renderer},
        }
        if self.renderer == "remote":
            result["backends"] = [
                {
                    "url": "https://ovrtx.example.test",
                    "engine": "ovrtx",
                    "protocol_version": 2,
                    "status": "ready",
                }
            ]
        return result

    def render_view(
        self, *, output_dir: Path, name: str, direction: str, **_kwargs: object
    ) -> dict[str, object]:
        self.render_timeouts.append(float(_kwargs["timeout_seconds"]))
        self._advance_checkpoint(f"render_view:{name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        image = output_dir / f"{name}.png"
        color: str | tuple[int, int, int] = "black"
        if name.startswith("final_turntable_"):
            index = int(name.rsplit("_", 1)[-1])
            color = (index * 9 % 256, index * 17 % 256, index * 29 % 256)
        Image.new("RGB", (640, 480), color).save(image)
        response = output_dir / f"{name}_response.json"
        camera = output_dir / f"{name}_camera.json"
        response.write_text(
            json.dumps({"ok": True, "summary": {"backend": self.renderer}}) + "\n",
            encoding="utf-8",
        )
        camera.write_text("{}\n", encoding="utf-8")
        return {
            "name": name,
            "direction": direction,
            "image_path": str(image),
            "response_path": str(response),
            "camera_json_path": str(camera),
            "renderer": self.renderer,
        }

    def run_json(self, arguments: list[str], **_kwargs: object) -> dict[str, object]:
        self._advance_checkpoint(" ".join(arguments))
        self.commands.append(arguments)
        timeout = _kwargs.get("timeout_seconds")
        self.command_timeouts.append(float(timeout) if timeout is not None else None)
        if arguments == ["appearance", "audit"]:
            return {
                "ok": True,
                "data": {
                    "clear": True,
                    "counts": {
                        "effective_material_bindings": 0,
                        "effective_shader_appearances": 0,
                        "direct_shader_outputs": 0,
                        "display_values": 0,
                    },
                },
            }
        if arguments == ["material", "audit", "--effective", "--include-subsets"]:
            return self.material_audit_response
        if arguments[0] == "material-binding":
            prim_path = arguments[1]
            material_name = self.bound_materials.get(prim_path)
            bound_path = self.bound_material_paths.get(
                prim_path,
                (
                    f"/Looks/{Tf.MakeValidIdentifier(material_name)}"
                    if material_name is not None
                    else None
                ),
            )
            if prim_path in self.binding_overrides:
                bound_path = self.binding_overrides[prim_path]
            return {
                "ok": True,
                "data": {"bound_material_path": bound_path},
            }
        if arguments[0] == "material":
            prim_path = arguments[1]
            if "--library-prim" in arguments:
                source_path = arguments[arguments.index("--library-prim") + 1]
                self.bound_materials[prim_path] = source_path.rsplit("/", 1)[-1]
                local_path = f"/Looks/{source_path.strip('/').replace('/', '__')}"
                self.bound_material_paths[prim_path] = local_path
            else:
                self.bound_materials[prim_path] = arguments[
                    arguments.index("--name") + 1
                ]
                local_path = (
                    f"/Looks/{Tf.MakeValidIdentifier(self.bound_materials[prim_path])}"
                )
                self.bound_material_paths[prim_path] = local_path
            return {
                "ok": True,
                "data": {"material_path": local_path},
            }
        if arguments[0] == "save":
            output = Path(arguments[1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("#usda 1.0\n", encoding="utf-8")
        if arguments[0] == "render" and "--seg" in arguments:
            output = Path(arguments[arguments.index("--output") + 1])
            output_dir = output.parent if output.suffix else output
            output_dir.mkdir(parents=True, exist_ok=True)
            segmentation = (
                output
                if output.suffix
                else output_dir / f"segmentation-{len(self.commands)}.png"
            )
            Image.new("RGB", (640, 480), (9, 8, 7)).save(segmentation)
            legend = segmentation.with_suffix(".legend.txt")
            legend.write_text("/Cabinet\trgb(9,8,7)\n", encoding="utf-8")
            return {
                "ok": True,
                "artifacts": [
                    {"label": "segmentation", "path": str(segmentation)},
                    {"label": "segmentation legend", "path": str(legend)},
                ],
            }
        return {
            "ok": True,
            "artifacts": [],
            "summary": {"backend": self.renderer},
        }

    def close(self) -> None:
        self.closed += 1


def _fake_evidence(**kwargs: object) -> dict[str, str]:
    run_dir = Path(kwargs["run_dir"])
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    paths = {
        "visible_candidates": raw / "visible_candidate_prims.json",
        "material_palette": raw / "material_palette.json",
        "material_authoring_context": raw / "material_authoring_context.json",
        "material_authoring_context_md": raw / "material_authoring_context.md",
        "material_assignment_seed": raw / "material_assignment_seed.json",
        "visible_candidate_table": raw / "visible_candidate_table.tsv",
    }
    paths["visible_candidates"].write_text(
        json.dumps(
            {
                "schema_version": "content-agents.visible-candidate-prims.v1",
                "path_space": "source",
                "candidate_visible_prim_count": 0,
                "candidates": [],
            }
        ),
        encoding="utf-8",
    )
    paths["material_palette"].write_text("{}\n", encoding="utf-8")
    paths["material_authoring_context"].write_text("{}\n", encoding="utf-8")
    paths["material_authoring_context_md"].write_text(
        "# Material context\n", encoding="utf-8"
    )
    paths["material_assignment_seed"].write_text("{}\n", encoding="utf-8")
    paths["visible_candidate_table"].write_text("source_path\n", encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[MaterialAssignConfig, _UsdCliSession]:
    config = _config(tmp_path)
    assert config.output_dir is not None
    session = _UsdCliSession(config.output_dir)

    def stage_inputs(**kwargs: object) -> None:
        run_dir = Path(kwargs["run_dir"])
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for label, source in (
            ("source", config.usd_path),
            ("material_library", config.materials_usd),
        ):
            staged = run_dir / "inputs" / label / source.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(source.read_bytes())
            (raw_dir / f"staged_input_{label}.json").write_text(
                json.dumps({"staged_usd_path": str(staged)}),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        material_coordinator,
        "_session",
        lambda *_args, **_kwargs: session,
    )
    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", _fake_evidence
    )
    monkeypatch.setattr(
        material_coordinator,
        "_prepare_usd_cli_material_inputs",
        stage_inputs,
    )
    return config, session


def _initial_checked_views(run_dir: Path) -> list[str]:
    preparation = material_coordinator.MaterialCoordinatorPreparation.model_validate(
        json.loads((run_dir / "coordinator_preparation.json").read_text())
    )
    request = material_coordinator.MaterialCoordinatorRequest.model_validate(
        json.loads((run_dir / "coordinator_request.json").read_text())
    )
    return [
        binding.path
        for binding in (
            *preparation.initial_render_bindings,
            *request.reference_images,
            *request.reference_files,
        )
    ]


def _write_valid_decision(run_dir: Path) -> Path:
    decision = run_dir / "raw" / "material_decision_patch.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "candidate_count": 0,
                "material_assignments": [],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "pass",
                    "checked_views": _initial_checked_views(run_dir),
                    "unresolved_issues": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return decision


def _write_post_apply_review(
    run_dir: Path,
    receipt: material_coordinator.MaterialCoordinatorReviewRequired,
    *,
    status: str = "pass",
    unresolved_issues: list[str] | None = None,
) -> Path:
    review = run_dir / "raw" / "material_post_apply_review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-post-apply-review.v1",
                "status": status,
                "checked_views": [
                    binding.path for binding in receipt.final_render_bindings
                ],
                "checked_view_bindings": [
                    binding.model_dump(mode="json")
                    for binding in receipt.final_render_bindings
                ],
                "issues_found": [],
                "issues_fixed": [],
                "unresolved_issues": unresolved_issues or [],
                "assessment_notes": "Reviewed the exact post-apply OVRTX renders.",
            }
        ),
        encoding="utf-8",
    )
    return review


@pytest.mark.parametrize(
    ("suffix", "expected_format"),
    ((".usda", "usda"), (".usdc", "usdc"), (".usdz", None)),
)
def test_publish_materialized_usd_uses_requested_format(
    tmp_path: Path,
    suffix: str,
    expected_format: str | None,
) -> None:
    Usd = pytest.importorskip("pxr.Usd")
    source = tmp_path / "run" / "materialized.usda"
    source.parent.mkdir()
    source.write_text(
        '#usda 1.0\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    output = tmp_path / "published" / f"materialized{suffix}"

    material_coordinator._publish_materialized_usd(source, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetPrimAtPath("/World")
    if expected_format is None:
        assert zipfile.is_zipfile(output)
    else:
        assert stage.GetRootLayer().GetFileFormat().formatId == expected_format


def test_prepare_uses_one_usd_cli_session_and_seals_ovrtx_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "_run_child_agent",
        lambda **_kwargs: pytest.fail("Material prepare launched a nested agent"),
    )

    result = prepare_material_for_coordinator(config)

    assert result.session_id == session.session_id
    assert len(result.initial_render_paths) == 3
    assert len(result.initial_render_bindings) == 18
    assert Path(result.initial_render_records_binding.path).is_file()
    assert Path(result.receipt_checkpoint_binding.path).is_file()
    packet = json.loads(Path(result.packet_path).read_text(encoding="utf-8"))
    assert packet["scene_tool"] == "usd-cli"
    assert all(
        "segmentation_path" in record for record in packet["initial_evidence_renders"]
    )
    assert Path(result.authoring_context_path).is_file()
    assert Path(result.assignment_seed_path).is_file()
    assert session.opened == [
        config.output_dir / "inputs" / "source" / config.usd_path.name
    ]
    assert session.closed == 0
    assert session.render_timeouts == [300.0, 300.0, 300.0]
    assert [
        timeout
        for command, timeout in zip(
            session.commands, session.command_timeouts, strict=True
        )
        if command[:2] == ["render", "--seg"]
    ] == [300.0, 300.0, 300.0]


def test_prepare_honors_a_larger_workflow_render_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    config = replace(config, scene_tool_timeout_seconds=420.0)

    prepare_material_for_coordinator(config)

    assert session.render_timeouts == [420.0, 420.0, 420.0]


@pytest.mark.parametrize("failure_stage", ("probe", "render"))
def test_prepare_failure_closes_unreceipted_session_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)

    def fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"injected {failure_stage} failure")

    monkeypatch.setattr(
        session,
        "require_ovrtx" if failure_stage == "probe" else "render_view",
        fail,
    )

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        prepare_material_for_coordinator(config)

    assert session.closed == 1
    assert config.output_dir is not None
    assert not (config.output_dir / "coordinator_preparation.json").exists()


@pytest.mark.parametrize(
    "artifact_kind",
    ("segmentation", "segmentation_response", "render_records"),
)
def test_finalize_rejects_tampered_initial_decision_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_kind: str,
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    preparation = prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = _write_valid_decision(config.output_dir)
    if artifact_kind == "render_records":
        target = Path(preparation.initial_render_records_binding.path)
    else:
        suffix = (
            "_segmentation.json"
            if artifact_kind == "segmentation_response"
            else "_segmentation.png"
        )
        target = next(
            Path(binding.path)
            for binding in preparation.initial_render_bindings
            if binding.path.endswith(suffix)
        )
    target.write_bytes(target.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="evidence identity changed"):
        finalize_material_for_coordinator(
            config.output_dir,
            decision_patch_path=decision,
        )
    assert not any(command[0] == "material" for command in session.commands)


def test_finalize_rejects_arbitrary_child_created_checked_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision_path = _write_valid_decision(config.output_dir)
    arbitrary = config.output_dir / "raw" / "child_claimed_review.png"
    Image.new("RGB", (64, 64), "red").save(arbitrary)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["visual_quality_assessment"]["checked_views"] = [str(arbitrary)]
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(ValueError, match="exact sealed initial OVRTX"):
        finalize_material_for_coordinator(
            config.output_dir,
            decision_patch_path=decision_path,
        )
    assert not any(command[0] == "material" for command in session.commands)


def test_prepare_uses_workflow_optimizer_inspection_and_correspondence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    config = replace(
        config,
        optimize=True,
        material_candidate_space="inspection",
    )
    captured: dict[str, object] = {}

    class Optimization:
        def __init__(self, run_dir: Path) -> None:
            optimizer_dir = run_dir / "raw" / "scene_optimizer"
            optimizer_dir.mkdir(parents=True, exist_ok=True)
            self.inspection_usd_path = optimizer_dir / "inspection.usda"
            self.inspection_usd_path.write_text(
                '#usda 1.0\ndef Xform "Optimized" {}\n', encoding="utf-8"
            )
            self.correspondence_path = optimizer_dir / "correspondence.json"
            self.correspondence_path.write_text(
                json.dumps(
                    {
                        "source_to_inspection": {"/Cabinet": ["/Optimized"]},
                        "inspection_to_source": {"/Optimized": ["/Cabinet"]},
                    }
                ),
                encoding="utf-8",
            )

        def prompt_metadata(self) -> dict[str, object]:
            return {
                "enabled": True,
                "inspection_usd_path": str(self.inspection_usd_path),
            }

    def prepare_optimized(**kwargs: object) -> Optimization:
        captured["optimizer_config"] = kwargs["config"]
        return Optimization(Path(kwargs["run_dir"]))

    def evidence_with_capture(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _fake_evidence(**kwargs)

    monkeypatch.setattr(
        material_coordinator,
        "_prepare_usd_cli_optimized_inspection",
        prepare_optimized,
    )
    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", evidence_with_capture
    )

    result = prepare_material_for_coordinator(config)

    inspection = Path(captured["inspection_usd"])
    assert captured["optimizer_config"] is config
    assert session.opened == [inspection]
    assert captured["source_usd"] == config.usd_path.resolve()
    assert captured["correspondence"].translate_inspection_to_source(
        "/Optimized"
    ).source_paths == ["/Cabinet"]
    packet = json.loads(Path(result.packet_path).read_text(encoding="utf-8"))
    assert packet["inspection_usd"] == str(inspection)
    assert packet["scene_optimizer"]["enabled"] is True


def test_prepare_cli_discovers_repo_from_usd_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    code = material_coordinator.main(
        [
            "prepare",
            "--usd",
            str(config.usd_path),
            "--materials-yaml",
            str(config.materials_yaml),
            "--materials-usd",
            str(config.materials_usd),
            "--output-dir",
            str(config.output_dir),
            "--output-usd",
            str(config.output_usd_path),
            "--reference-image",
            str(config.reference_images[0]),
            "--no-optimize",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    request = json.loads(Path(payload["request_path"]).read_text(encoding="utf-8"))
    assert request["repository_root"] == str(config.repo_root)
    assert request["enable_deduplicate"] is False


def test_material_coordinator_cli_writes_failures_only_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        material_coordinator.main(
            [
                "release",
                "--run-dir",
                str(tmp_path / "missing"),
                "--preparation-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_prepare_rejects_reusing_a_coordinator_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    preparation = prepare_material_for_coordinator(config)
    with pytest.raises(ValueError, match="fresh run"):
        prepare_material_for_coordinator(config)
    assert Path(preparation.request_path).is_file()


def test_release_closes_the_single_prepared_usd_cli_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    result = release_material_for_coordinator(config.output_dir)
    assert result.status == "released"
    assert result.scene_tool == "usd-cli"
    assert session.closed == 1


def test_release_closes_review_required_session_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )
    assert receipt.status == "review_required"
    assert session.render_timeouts == [300.0] * 31
    assert session.closed == 0

    result = release_material_for_coordinator(config.output_dir)

    assert result.status == "released"
    assert session.closed == 1
    assert not config.output_usd_path.is_file()
    assert not (config.output_dir / "coordinator_result.json").exists()


def test_finalize_rejects_missing_vqa_and_preserves_sealed_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = config.output_dir / "raw" / "material_decision_patch.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "material_assignments": [],
                "reviewed_no_override": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="visual_quality_assessment"):
        finalize_material_for_coordinator(
            config.output_dir, decision_patch_path=decision
        )
    assert session.closed == 0


def test_finalize_rejects_input_and_evidence_drift_before_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = _write_valid_decision(config.output_dir)
    config.usd_path.write_text('#usda 1.0\ndef Xform "Changed" {}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        finalize_material_for_coordinator(
            config.output_dir, decision_patch_path=decision
        )
    assert not any(command[0] == "material" for command in session.commands)


def test_finalize_rejects_audited_material_that_differs_from_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None

    def evidence_with_body(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": 1,
                    "candidates": [
                        {
                            "source_path": "/Cabinet",
                            "prim_path": "/Cabinet",
                            "shape_hints": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", evidence_with_body
    )
    prepare_material_for_coordinator(config)
    decision = config.output_dir / "raw" / "material_decision_patch.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "candidate_count": 1,
                "material_assignments": [
                    {
                        "family": "cabinet",
                        "material_name": "paint",
                        "material_path": "/Materials/Paint",
                        "source_prim_paths": ["/Cabinet"],
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "pass",
                    "checked_views": _initial_checked_views(config.output_dir),
                    "unresolved_issues": [],
                },
            }
        ),
        encoding="utf-8",
    )
    session.material_audit_response = {
        "ok": True,
        "data": {
            "counts": {"invalid_binding_targets": 0},
            "renderables": [{"path": "/Cabinet", "material": "/Other/Paint"}],
        },
    }
    # A basename-only verifier would accept this unrelated Material.  The
    # coordinator must require the exact local identity returned by usd-cli.
    session.binding_overrides["/Cabinet"] = "/Other/Paint"

    with pytest.raises(ValueError, match="expected chosen material"):
        finalize_material_for_coordinator(
            config.output_dir, decision_patch_path=decision
        )
    assert not (config.output_dir / "raw" / "material_binding_audit.json").exists()


def test_finalize_imports_and_verifies_exact_manifest_material_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The display label never replaces the manifest's exact library prim path."""

    config, session = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None
    config.materials_yaml.write_text(
        "entries:\n  - name: Paint Red\n    binding: /World/Looks/Red\n",
        encoding="utf-8",
    )
    config.materials_usd.write_text(
        "#usda 1.0\n\n"
        'def Xform "World"\n'
        "{\n"
        '    def Scope "Looks"\n'
        "    {\n"
        '        def Material "Red"\n'
        "        {\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )

    def evidence_with_body(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": 1,
                    "candidates": [
                        {
                            "source_path": "/Cabinet",
                            "prim_path": "/Cabinet",
                            "shape_hints": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", evidence_with_body
    )
    prepare_material_for_coordinator(config)
    decision = config.output_dir / "raw" / "material_decision_patch.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "content-agents.material-decision-patch.v1",
                "candidate_count": 1,
                "material_assignments": [
                    {
                        "family": "cabinet",
                        "material_name": "Paint Red",
                        "material_path": "/World/Looks/Red",
                        "source_prim_paths": ["/Cabinet"],
                    }
                ],
                "reviewed_no_override": [],
                "visual_quality_assessment": {
                    "status": "pass",
                    "checked_views": _initial_checked_views(config.output_dir),
                    "unresolved_issues": [],
                },
            }
        ),
        encoding="utf-8",
    )

    finalize_material_for_coordinator(config.output_dir, decision_patch_path=decision)

    audit = json.loads(
        (config.output_dir / "raw" / "material_binding_audit.json").read_text()
    )
    assert audit["records"][0]["material_name"] == "Paint Red"
    assert audit["records"][0]["material_source_path"] == "/World/Looks/Red"
    assert audit["records"][0]["bound_material_path"] == "/Looks/World__Looks__Red"
    assert [
        "material",
        "/Cabinet",
        "--library",
        str(config.output_dir / "inputs" / "material_library" / "materials.usda"),
        "--library-prim",
        "/World/Looks/Red",
    ] in session.commands


def test_finalize_rejects_policy_invalid_paths_before_usd_cli_authoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = _write_valid_decision(config.output_dir)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["material_assignments"] = [
        {
            "material_name": "paint",
            "material_path": "/Materials/Paint",
            "source_prim_paths": ["/not-a-candidate"],
        }
    ]
    decision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate"):
        finalize_material_for_coordinator(
            config.output_dir, decision_patch_path=decision
        )
    rejected = config.output_dir / "raw" / "rejected_material_assignments.json"
    assert rejected.is_file()
    assert not any(command[0] == "material" for command in session.commands)


def test_finalize_authors_workflow_normalized_source_alias_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    assert config.output_dir is not None
    session = _UsdCliSession(config.output_dir)
    monkeypatch.setattr(
        material_coordinator,
        "_session",
        lambda *_args, **_kwargs: session,
    )
    monkeypatch.setattr(
        material_coordinator,
        "_staged_scene_inputs",
        lambda *_args: (config.usd_path, config.materials_usd),
    )

    def evidence(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": 1,
                    "candidates": [
                        {
                            "source_path": "/Cabinet/Panel",
                            "runtime_path": "/Optimized/Panel",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", evidence
    )
    monkeypatch.setattr(
        material_coordinator,
        "_prepare_usd_cli_material_inputs",
        lambda **_kwargs: None,
    )
    prepare_material_for_coordinator(config)
    decision = _write_valid_decision(config.output_dir)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["material_assignments"] = [
        {
            "material_name": "paint",
            "material_path": "/Materials/Paint",
            "prim_paths": ["/Optimized/Panel"],
        }
    ]
    decision.write_text(json.dumps(payload), encoding="utf-8")
    session.material_audit_response = {
        "ok": True,
        "data": {
            "counts": {"invalid_binding_targets": 0},
            "renderables": [{"path": "/Cabinet/Panel", "material": "/Looks/paint"}],
        },
    }

    receipt = finalize_material_for_coordinator(
        config.output_dir, decision_patch_path=decision
    )
    review_material_for_coordinator(
        config.output_dir,
        review_patch_path=_write_post_apply_review(config.output_dir, receipt),
    )

    material_commands = [
        command
        for command in session.commands
        if command[0] == "material" and "--library" in command
    ]
    assert material_commands[0][1] == "/Cabinet/Panel"
    applied = json.loads(
        (config.output_dir / "raw" / "material_applied_decision_patch.json").read_text()
    )
    assert applied["material_assignments"][0]["prim_paths"] == ["/Cabinet/Panel"]
    assignments = json.loads((config.output_dir / "assignments.json").read_text())
    assert assignments["coverage"] == {
        "candidate_visible_prim_count": 1,
        "material_assignment_prim_count": 1,
        "reviewed_no_override_prim_count": 0,
        "claimed_candidate_prim_count": 1,
        "unassigned_visible_prim_count": 0,
        "material_decision_prim_count": 1,
        "missing_assignment_prim_count": 0,
        "rejected_assignment_prim_count": 0,
    }


def test_finalize_deinstances_external_reference_before_distinct_part_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    config = replace(config, respect_existing_material_bindings=True)
    assert config.output_dir is not None
    config.materials_yaml.write_text(
        "entries:\n"
        "  - name: paint\n    binding: /Materials/Paint\n"
        "  - name: trim\n    binding: /Materials/Trim\n",
        encoding="utf-8",
    )
    config.materials_usd.write_text(
        '#usda 1.0\ndef Scope "Materials" {\n'
        '    def Material "Paint" {}\n'
        '    def Material "Trim" {}\n'
        "}\n",
        encoding="utf-8",
    )

    def external_evidence(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": 2,
                    "candidates": [
                        {
                            "source_path": "/ExternalCopy/Body",
                            "runtime_path": "/ExternalCopy/Body",
                            "deinstance_root_paths": ["/ExternalCopy"],
                        },
                        {
                            "source_path": "/ExternalCopy/Trim",
                            "runtime_path": "/ExternalCopy/Trim",
                            "deinstance_root_paths": ["/ExternalCopy"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator,
        "build_material_authoring_evidence",
        external_evidence,
    )
    prepare_material_for_coordinator(config)
    decision = _write_valid_decision(config.output_dir)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["material_assignments"] = [
        {
            "family": "body",
            "material_name": "paint",
            "material_path": "/Materials/Paint",
            "source_prim_paths": ["/ExternalCopy/Body"],
        },
        {
            "family": "trim",
            "material_name": "trim",
            "material_path": "/Materials/Trim",
            "source_prim_paths": ["/ExternalCopy/Trim"],
        },
    ]
    decision.write_text(json.dumps(payload), encoding="utf-8")

    finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=decision,
    )

    deinstance_index = session.commands.index(
        ["set", "/ExternalCopy", "instanceable", "false"]
    )
    material_commands = [
        command
        for command in session.commands
        if command[0] == "material" and "--library" in command
    ]
    assert deinstance_index < session.commands.index(material_commands[0])
    assert [command[1] for command in material_commands] == [
        "/ExternalCopy/Body",
        "/ExternalCopy/Trim",
    ]
    assert session.bound_materials == {
        "/ExternalCopy/Body": "Paint",
        "/ExternalCopy/Trim": "Trim",
    }
    assert ["appearance", "clear"] not in session.commands


def test_finalize_verifies_all_targets_when_aggregate_audit_is_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None
    target_paths = [f"/Cabinet/Part_{index:03d}" for index in range(205)]

    def large_evidence(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": len(target_paths),
                    "candidates": [
                        {"source_path": path, "prim_path": path, "shape_hints": []}
                        for path in target_paths
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", large_evidence
    )
    prepare_material_for_coordinator(config)
    decision = _write_valid_decision(config.output_dir)
    payload = json.loads(decision.read_text())
    payload["candidate_count"] = len(target_paths)
    payload["material_assignments"] = [
        {
            "family": f"parts-{index // 16}",
            "material_name": "paint",
            "material_path": "/Materials/Paint",
            "source_prim_paths": target_paths[index : index + 16],
        }
        for index in range(0, len(target_paths), 16)
    ]
    decision.write_text(json.dumps(payload), encoding="utf-8")
    session.material_audit_response = {
        "ok": True,
        "data": {
            "counts": {
                "renderables": len(target_paths),
                "invalid_binding_targets": 0,
            },
            "renderables": [
                {"path": path, "material": "/Looks/paint"} for path in target_paths[:50]
            ],
            "renderables_omitted": len(target_paths) - 50,
            "truncated": True,
        },
    }

    finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=decision,
    )

    audit = json.loads(
        (config.output_dir / "raw" / "material_binding_audit.json").read_text()
    )
    assert audit["expected_target_count"] == len(target_paths)
    assert audit["verified_target_count"] == len(target_paths)
    assert audit["aggregate_audit_truncated"] is True
    assert audit["aggregate_omitted_counts"] == {"renderables_omitted": 155}
    assert len(
        [command for command in session.commands if command[0] == "material-binding"]
    ) == len(target_paths)


def test_finalize_verifies_selected_geomsubset_with_neutral_binding_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None
    subset_path = "/Cabinet/Panel/materialBind/FrontFaces"

    def subset_evidence(**kwargs: object) -> dict[str, str]:
        paths = _fake_evidence(**kwargs)
        Path(paths["visible_candidates"]).write_text(
            json.dumps(
                {
                    "schema_version": "content-agents.visible-candidate-prims.v1",
                    "path_space": "source",
                    "candidate_visible_prim_count": 1,
                    "candidates": [
                        {
                            "source_path": subset_path,
                            "prim_path": subset_path,
                            "shape_hints": ["geom_subset"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return paths

    monkeypatch.setattr(
        material_coordinator, "build_material_authoring_evidence", subset_evidence
    )
    prepare_material_for_coordinator(config)
    decision = _write_valid_decision(config.output_dir)
    payload = json.loads(decision.read_text())
    payload["candidate_count"] = 1
    payload["material_assignments"] = [
        {
            "family": "front faces",
            "material_name": "paint",
            "material_path": "/Materials/Paint",
            "source_prim_paths": [subset_path],
        }
    ]
    decision.write_text(json.dumps(payload), encoding="utf-8")

    finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=decision,
    )

    assert ["material-binding", subset_path] in session.commands
    audit = json.loads(
        (config.output_dir / "raw" / "material_binding_audit.json").read_text()
    )
    assert audit["records"] == [
        {
            "prim_path": subset_path,
            "material_name": "paint",
            "material_source_path": "/Materials/Paint",
            "bound_material_path": "/Looks/Materials__Paint",
            "status": "pass",
            "query_response": {
                "ok": True,
                "data": {"bound_material_path": "/Looks/Materials__Paint"},
            },
        }
    ]


def test_finalize_writes_policy_and_ovrtx_final_segmentation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = _write_valid_decision(config.output_dir)
    receipt = finalize_material_for_coordinator(
        config.output_dir, decision_patch_path=decision
    )
    assert receipt.status == "review_required"
    assert receipt.unresolved_issues == [
        "Post-apply OVRTX renders require an evidence-bound visual review."
    ]
    assert session.closed == 0
    assert not config.output_usd_path.is_file()
    assert not (config.output_dir / "coordinator_result.json").exists()
    assert not (config.output_dir / "visual_quality_assessment.json").exists()
    policy = json.loads(
        (config.output_dir / "raw" / "material_finalization_policy.json").read_text()
    )
    assert policy["coverage"]["candidate_visible_prim_count"] == 0
    final = json.loads(
        (config.output_dir / "raw" / "final_render_records.json").read_text()
    )
    assert len(final["renders"]) == 28
    assert final["turntable"]["frame_count"] == 24
    turntable = Path(final["turntable"]["gif_path"])
    assert turntable.is_file()
    assert str(turntable.resolve()) in {item.path for item in receipt.evidence}
    with Image.open(turntable) as animation:
        assert animation.n_frames == 24
    evidence_paths = {item.path for item in receipt.evidence}
    cleared_usd = config.output_dir / "raw" / "appearance_cleared.usda"
    materialized_usd = config.output_dir / "output" / "materialized.usda"
    assert cleared_usd.is_file()
    assert str(cleared_usd.resolve()) in evidence_paths
    assert session.opened[-2:] == [cleared_usd, materialized_usd]
    rendered_usd = material_coordinator._bound(materialized_usd)
    _images, _paths, validation_artifacts = material_coordinator._final_render_evidence(
        run_dir=config.output_dir,
        records=tuple(final["renders"]),
    )
    render_evidence = {
        item.path: item for item in validation_artifacts if item.kind == "render"
    }
    for item in final["renders"]:
        assert item["renderer"] == "ovrtx"
        assert item["rendered_usd"] == rendered_usd.model_dump(mode="json")
        required_keys = [
            "image_path",
            "response_path",
            "camera_json_path",
        ]
        if item["kind"] == "verification_view":
            required_keys.extend(
                [
                    "segmentation_path",
                    "segmentation_legend_path",
                    "segmentation_response_path",
                ]
            )
        for key in required_keys:
            assert Path(item[key]).is_file()
            assert str(Path(item[key]).resolve()) in evidence_paths
        response = json.loads(Path(item["response_path"]).read_text())
        assert response["summary"]["backend"] == "ovrtx"
        if item["kind"] == "verification_view":
            render_metadata = render_evidence[str(Path(item["image_path"]).resolve())]
            assert render_metadata.metadata["rendered_usd_path"] == rendered_usd.path
            assert (
                render_metadata.metadata["rendered_usd_sha256"] == rendered_usd.sha256
            )
        if item["kind"] == "verification_view":
            segmentation_response = json.loads(
                Path(item["segmentation_response_path"]).read_text()
            )
            assert segmentation_response == item["segmentation_response"]

    result = review_material_for_coordinator(
        config.output_dir,
        review_patch_path=_write_post_apply_review(config.output_dir, receipt),
    )
    assert result.status == "pass"
    assert session.closed == 1
    assessment = json.loads(
        (config.output_dir / "visual_quality_assessment.json").read_text()
    )
    assert assessment["status"] == "pass"
    assert assessment["checked_views"] == [
        binding.path for binding in receipt.final_render_bindings
    ]
    assert assessment["final_render_review"]["status"] == "reviewed"
    assert runner._valid_usd_cli_visual_quality(
        assessment,
        run_dir=config.output_dir,
    )
    validation = json.loads(
        (config.output_dir / "validation_evidence.json").read_text()
    )
    assert validation["sim_ready_status"] == "pass"
    assert validation["checks"][0]["status"] == "pass"


def test_review_rejects_post_apply_render_drift_without_closing_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )
    review = _write_post_apply_review(config.output_dir, receipt)
    Path(receipt.final_render_bindings[0].path).write_bytes(b"drifted")

    with pytest.raises(ValueError, match="identity changed"):
        review_material_for_coordinator(
            config.output_dir,
            review_patch_path=review,
        )
    assert session.closed == 0
    assert not config.output_usd_path.is_file()


def test_finalize_never_reapplies_an_existing_review_required_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    decision = _write_valid_decision(config.output_dir)
    finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=decision,
    )
    command_count = len(session.commands)

    with pytest.raises(ValueError, match="already applied"):
        finalize_material_for_coordinator(
            config.output_dir,
            decision_patch_path=decision,
        )

    assert len(session.commands) == command_count
    assert session.closed == 0


def test_review_retains_post_apply_unresolved_vqa_status_and_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )
    result = review_material_for_coordinator(
        config.output_dir,
        review_patch_path=_write_post_apply_review(
            config.output_dir,
            receipt,
            status="unresolved_issues",
            unresolved_issues=["The final side view is too dark."],
        ),
    )

    assert result.status == "conditional"
    assert result.unresolved_issues == ["The final side view is too dark."]
    assessment = json.loads(
        (config.output_dir / "visual_quality_assessment.json").read_text()
    )
    assert assessment["status"] == "unresolved_issues"
    assert assessment["unresolved_issues"] == result.unresolved_issues


def test_remote_ovrtx_transport_can_complete_post_apply_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    session.renderer = "remote"
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )

    result = review_material_for_coordinator(
        config.output_dir,
        review_patch_path=_write_post_apply_review(config.output_dir, receipt),
    )

    assert result.status == "pass"
    probe = json.loads(
        (config.output_dir / "raw" / "ovrtx_render_probe.json").read_text()
    )
    assert probe["engine"] == "ovrtx"
    assert probe["transport"] == "remote"
    records = json.loads(
        (config.output_dir / "raw" / "final_render_records.json").read_text()
    )
    assert records["render_engine"] == "ovrtx"
    assert records["transports"] == ["remote"]
    assert all(record["renderer"] == "remote" for record in records["renders"])


def test_final_render_segmentation_integrates_with_workflow_grounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    finalize_material_for_coordinator(
        config.output_dir, decision_patch_path=_write_valid_decision(config.output_dir)
    )
    result = write_material_grounding_diagnostics(
        run_dir=config.output_dir,
        validation_iteration=1,
        unresolved_issues=["dark cabinet"],
    )
    assert result is not None
    diagnostics = json.loads(Path(result["aggregate"]).read_text())
    assert diagnostics["latest"]["operation_counts"]["pick_calls"] > 0
    assert diagnostics["latest"]["issues"][0]["views"][0]["picked_source_paths"] == [
        "/Cabinet"
    ]


def test_prepare_finalize_review_resume_with_fresh_session_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _shared = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None
    sessions: list[_UsdCliSession] = []
    resumed_digests: list[str | None] = []

    def fresh_session(
        *_args: object,
        receipt_checkpoint_sha256: str | None = None,
        **_kwargs: object,
    ) -> _UsdCliSession:
        checkpoint = (
            config.output_dir / "raw" / "usd_cli_command_receipts.checkpoint.json"
        )
        if receipt_checkpoint_sha256 is not None:
            assert _manifest_sha256(checkpoint) == receipt_checkpoint_sha256
        resumed_digests.append(receipt_checkpoint_sha256)
        session = _UsdCliSession(config.output_dir)
        sessions.append(session)
        return session

    monkeypatch.setattr(material_coordinator, "_session", fresh_session)
    preparation = prepare_material_for_coordinator(config)
    prepare_checkpoint = preparation.receipt_checkpoint_binding.sha256
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )
    application_checkpoint = receipt.receipt_checkpoint_binding.sha256
    review_material_for_coordinator(
        config.output_dir,
        review_patch_path=_write_post_apply_review(config.output_dir, receipt),
    )

    assert len(sessions) == 3
    assert len({id(session) for session in sessions}) == 3
    assert resumed_digests == [None, prepare_checkpoint, application_checkpoint]
    assert application_checkpoint != prepare_checkpoint
    assert sessions[1].open_force_reloads[0] is True
    assert sessions[-1].closed == 1


def test_finalize_rejects_rewritten_preparation_with_original_parent_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session = _prepare(tmp_path, monkeypatch)
    preparation = prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    preparation_path = config.output_dir / "coordinator_preparation.json"
    parent_digest = _manifest_sha256(preparation_path)
    checkpoint = Path(preparation.receipt_checkpoint_binding.path)
    checkpoint.write_text('{"attacker":"replacement"}\n', encoding="utf-8")
    rewritten = json.loads(preparation_path.read_text(encoding="utf-8"))
    rewritten["receipt_checkpoint_binding"] = material_coordinator._bound(
        checkpoint
    ).model_dump(mode="json")
    preparation_path.write_text(json.dumps(rewritten), encoding="utf-8")

    with pytest.raises(ValueError, match="parent phase boundary"):
        _finalize_material_for_coordinator(
            config.output_dir,
            decision_patch_path=_write_valid_decision(config.output_dir),
            preparation_sha256=parent_digest,
        )
    assert not any(command[0] == "material" for command in session.commands)


def test_review_rejects_rewritten_application_receipt_with_parent_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    prepare_material_for_coordinator(config)
    assert config.output_dir is not None
    preparation_digest = _manifest_sha256(
        config.output_dir / "coordinator_preparation.json"
    )
    receipt = finalize_material_for_coordinator(
        config.output_dir,
        decision_patch_path=_write_valid_decision(config.output_dir),
    )
    receipt_path = config.output_dir / "raw" / "material_application_receipt.json"
    parent_receipt_digest = _manifest_sha256(receipt_path)
    review = _write_post_apply_review(config.output_dir, receipt)
    checkpoint = Path(receipt.receipt_checkpoint_binding.path)
    checkpoint.write_text('{"attacker":"replacement"}\n', encoding="utf-8")
    rewritten = json.loads(receipt_path.read_text(encoding="utf-8"))
    rewritten["receipt_checkpoint_binding"] = material_coordinator._bound(
        checkpoint
    ).model_dump(mode="json")
    receipt_path.write_text(json.dumps(rewritten), encoding="utf-8")

    with pytest.raises(ValueError, match="parent phase boundary"):
        _review_material_for_coordinator(
            config.output_dir,
            review_patch_path=review,
            preparation_sha256=preparation_digest,
            application_receipt_sha256=parent_receipt_digest,
        )
    assert not config.output_usd_path.is_file()


def test_material_skill_carries_required_parent_phase_digests() -> None:
    skill = (
        Path(__file__).resolve().parents[3]
        / ".agents"
        / "skills"
        / "content-workflow-asset"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "literal `preparation_sha256` value from prepare" in skill
    assert "literal `application_receipt_sha256` value from" in skill
    assert "Do not recompute it from the child-writable run" in skill
    assert (
        '--preparation-sha256 "<literal preparation_sha256 returned by prepare>"'
        in skill
    )
    assert (
        '--application-receipt-sha256 "<literal application_receipt_sha256 returned by finalize>"'
        in skill
    )
    parsed = material_coordinator._build_parser().parse_args(
        [
            "review",
            "--run-dir",
            "/tmp/material-run",
            "--review-patch",
            "/tmp/material-run/raw/material_post_apply_review.json",
            "--preparation-sha256",
            "a" * 64,
            "--application-receipt-sha256",
            "b" * 64,
        ]
    )
    assert parsed.preparation_sha256 == "a" * 64
    assert parsed.application_receipt_sha256 == "b" * 64


def test_cli_phase_digests_survive_separate_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, _session = _prepare(tmp_path, monkeypatch)
    assert config.output_dir is not None
    prepare_args = [
        "prepare",
        "--usd",
        str(config.usd_path),
        "--materials-yaml",
        str(config.materials_yaml),
        "--materials-usd",
        str(config.materials_usd),
        "--output-dir",
        str(config.output_dir),
        "--output-usd",
        str(config.output_usd_path),
        "--reference-image",
        str(config.reference_images[0]),
        "--reference-file",
        str(config.reference_files[0]),
        "--no-optimize",
    ]
    assert material_coordinator.main(prepare_args) == 0
    prepare_stdout = json.loads(capsys.readouterr().out)
    preparation_sha256 = prepare_stdout["preparation_sha256"]
    assert preparation_sha256 == _manifest_sha256(
        config.output_dir / "coordinator_preparation.json"
    )

    decision = _write_valid_decision(config.output_dir)
    assert (
        material_coordinator.main(
            [
                "finalize",
                "--run-dir",
                str(config.output_dir),
                "--decision-patch",
                str(decision),
                "--preparation-sha256",
                preparation_sha256,
            ]
        )
        == 0
    )
    finalize_stdout = json.loads(capsys.readouterr().out)
    receipt_sha256 = finalize_stdout["application_receipt_sha256"]
    receipt = material_coordinator.MaterialCoordinatorReviewRequired.model_validate(
        {
            key: value
            for key, value in finalize_stdout.items()
            if key != "application_receipt_sha256"
        }
    )
    assert receipt_sha256 == _manifest_sha256(
        config.output_dir / "raw" / "material_application_receipt.json"
    )

    review = _write_post_apply_review(config.output_dir, receipt)
    assert (
        material_coordinator.main(
            [
                "review",
                "--run-dir",
                str(config.output_dir),
                "--review-patch",
                str(review),
                "--preparation-sha256",
                preparation_sha256,
                "--application-receipt-sha256",
                receipt_sha256,
            ]
        )
        == 0
    )
    review_stdout = json.loads(capsys.readouterr().out)
    assert review_stdout["status"] == "pass"
