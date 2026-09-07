# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from content_agent_workflows.asset_composition import (
    ArtifactBinding,
    AssetLeafProjection,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    CanonicalOvrtxEvidenceLeafInvocation,
    CanonicalOvrtxEvidenceLeafResult,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.artifacts import atomic_write_json
from content_agent_workflows.validation import (
    CanonicalVisualEvidencePublication,
    VerifiedOperationError,
    execution_artifact_binding,
    ingest_verified_operation_result,
    produce_canonical_visual_evidence,
)


def _assets(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.usda"
    output = tmp_path / "post_mutation.usda"
    dependency = tmp_path / "albedo.png"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output.write_text(
        '#usda 1.0\ndef Xform "Asset" (references = @albedo.png@) {}\n',
        encoding="utf-8",
    )
    dependency.write_bytes(b"texture-bytes")
    return source, output, dependency


def _install_fake_usd_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate_during_render: Path | None = None,
    mutate_on_close: Path | None = None,
    source_revision: str = "a" * 40,
    render_failure: Exception | None = None,
    close_failure: Exception | None = None,
    blank_suspect_views: set[str] | None = None,
) -> list[dict[str, Any]]:
    from content_agent_workflows.common import usd_cli_session

    calls: list[dict[str, Any]] = []

    class FakeSession:
        def __init__(self, project_dir: Path) -> None:
            self.project_dir = project_dir
            self.project_dir.mkdir(parents=True)
            raw = project_dir / "raw"
            raw.mkdir()
            self.receipt_file = raw / "usd_cli_command_receipts.jsonl"
            self.receipt_checkpoint_file = (
                raw / "usd_cli_command_receipts.checkpoint.json"
            )
            self.route = SimpleNamespace(source_revision=source_revision)
            self.session_id = "workflow-validation-test"
            self.workflow = "validation-canonical-visual-evidence"

        @classmethod
        def create(cls, **kwargs: Any) -> FakeSession:
            calls.append({"method": "create", **kwargs})
            return cls(Path(kwargs["project_dir"]))

        def require_ovrtx(self, output_dir: Path) -> dict[str, Any]:
            calls.append({"method": "require_ovrtx", "output_dir": output_dir})
            return {"schema_version": "usd-cli.render-probe.v1"}

        def open(self, scene_path: Path, *, read_only: bool = False) -> dict[str, Any]:
            calls.append(
                {
                    "method": "open",
                    "scene_path": scene_path,
                    "read_only": read_only,
                }
            )
            return {"ok": True}

        @staticmethod
        def stage_up_axis_is_y(_scene_path: Path) -> bool:
            return False

        def render_view(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({"method": "render_view", **kwargs})
            if render_failure is not None:
                raise render_failure
            if mutate_during_render is not None:
                mutate_during_render.write_text(
                    "changed during render", encoding="utf-8"
                )
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            name = str(kwargs["name"])
            image = output_dir / f"{name}.png"
            response = output_dir / f"{name}_response.json"
            camera = output_dir / f"{name}_camera.json"
            image.write_bytes(f"real-render-{name}".encode())
            response_payload: dict[str, Any] = {"ok": True}
            if name in (blank_suspect_views or set()):
                response_payload["data"] = {"blank_suspect": [str(image)]}
            response.write_text(json.dumps(response_payload), encoding="utf-8")
            camera.write_text(
                json.dumps({"direction": kwargs["direction"]}), encoding="utf-8"
            )
            return {
                "name": name,
                "direction": kwargs["direction"],
                "image_path": str(image),
                "response_path": str(response),
                "camera_json_path": str(camera),
                "renderer": kwargs["backend"],
                "renderer_identity": None,
            }

        def close(self) -> None:
            calls.append({"method": "close"})
            if mutate_on_close is not None:
                mutate_on_close.write_text(
                    "changed while closing session", encoding="utf-8"
                )
            self.receipt_file.write_text(
                json.dumps(
                    {
                        "tool": {
                            "name": "usd-cli",
                            "source_revision": self.route.source_revision,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.receipt_checkpoint_file.write_text(
                json.dumps({"receipt_sha256": "b" * 64}), encoding="utf-8"
            )
            if close_failure is not None:
                raise close_failure

    monkeypatch.setattr(usd_cli_session, "WorkflowUsdCliSession", FakeSession)
    return calls


def test_canonical_visual_leaf_binds_usd_cli_ovrtx_without_judging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    output_before_render = output.read_bytes()
    from world_understanding.validation import usd_rendering

    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    monkeypatch.setattr(
        usd_rendering,
        "render_usd_visual_evidence",
        lambda **_kwargs: pytest.fail("legacy renderer must not be called"),
    )
    monkeypatch.setenv("RENDER_ENDPOINT", "https://ambient-renderer.example.test")
    calls = _install_fake_usd_cli(monkeypatch)

    publication = produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / "visual",
        backend="ovrtx",
        views=("+x+y+z", "+x-y+z"),
    )

    renders = [call for call in calls if call["method"] == "render_view"]
    assert [call["direction"] for call in renders] == ["+x+y+z", "+x-y+z"]
    assert all(call["backend"] == "ovrtx" for call in renders)
    assert calls[0]["render_config"] == {
        "renderer": "ovrtx",
        "remote_url": "",
        "remote_api_key": "",
        "backends": [],
    }
    assert [call["method"] for call in calls[:3]] == [
        "create",
        "require_ovrtx",
        "open",
    ]
    assert calls[2]["read_only"] is False
    assert output.read_bytes() == output_before_render
    assert publication.result.output.sha256 == execution_artifact_binding(output).sha256
    assert publication.result.dependencies == (execution_artifact_binding(dependency),)
    assert publication.result.authority == "outer_review_input"
    assert publication.result.native_report_type == "usd-cli.ovrtx-render-report"
    assert (
        publication.result.native_payload_type
        == "visual.canonical-usd-cli-ovrtx-payload"
    )
    assert publication.result.tool is not None
    assert publication.result.tool.component_id == "package-owned-usd-cli-render"
    assert len(publication.result.artifacts) == 8
    for binding in publication.result.artifacts:
        assert binding.sha256 == execution_artifact_binding(binding.path).sha256

    runtime_binding = next(
        binding
        for binding in shared_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id == CANONICAL_OVRTX_EVIDENCE_LEAF_ID
    )
    invocation = CanonicalOvrtxEvidenceLeafInvocation(
        output_dir=str(tmp_path / "visual"),
        post_mutation_usd=str(output),
        source_usd=str(source),
        backend="ovrtx",
        views=("+x+y+z", "+x-y+z"),
        image_width=1024,
        image_height=1024,
    )
    invocation_path = tmp_path / "asset_leaf_invocation.json"
    atomic_write_json(invocation_path, invocation)

    def project(
        candidate: CanonicalVisualEvidencePublication,
        name: str,
    ) -> AssetLeafProjection:
        result_path = tmp_path / f"asset_leaf_result_{name}.json"
        atomic_write_json(result_path, candidate)
        invocation_artifact = ArtifactBinding.model_validate(
            execution_artifact_binding(invocation_path).model_dump(mode="json")
        )
        result_artifact = ArtifactBinding.model_validate(
            execution_artifact_binding(result_path).model_dump(mode="json")
        )
        return runtime_binding.project(
            invocation,
            CanonicalOvrtxEvidenceLeafResult.model_validate(
                candidate.model_dump(mode="json")
            ),
            invocation_artifact=invocation_artifact,
            result_artifact=result_artifact,
        )

    projected = project(publication, "pass")
    assert projected.payload.native_disposition == "passed"
    assert projected.payload.native_status == "pass"

    non_passing = publication.model_copy(
        update={
            "result": publication.result.model_copy(update={"native_status": "warn"})
        }
    )
    with pytest.raises(ValueError, match="native pass disposition"):
        project(non_passing, "warn")

    other_operation = publication.model_copy(
        update={
            "result": publication.result.model_copy(
                update={"operation_id": "visual.other-operation"}
            )
        }
    )
    with pytest.raises(ValueError, match="another operation or gate"):
        project(other_operation, "other_operation")

    substituted_chain = publication.model_copy(
        update={"payload": publication.render_report}
    )
    with pytest.raises(ValueError, match="artifact chain changed"):
        project(substituted_chain, "substituted_chain")

    receipt = ingest_verified_operation_result(
        publication.envelope.path,
        output_dir=tmp_path / "provided",
    )
    assert receipt.envelope == publication.result
    assert len([call for call in calls if call["method"] == "create"]) == 1


def test_canonical_visual_leaf_maps_environment_for_explicit_remote_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    monkeypatch.setenv("RENDER_ENDPOINT", "https://renderer.example.test")
    calls = _install_fake_usd_cli(monkeypatch)

    produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / "remote-visual",
        backend="remote",
        views=("+x+y+z",),
    )

    assert calls[0]["render_config"] == {
        "renderer": "remote",
        "remote_url": "https://renderer.example.test",
        "remote_api_key": "",
        "backends": [],
    }


@pytest.mark.parametrize("backend", ("ovrtx", "remote"))
def test_canonical_visual_leaf_reuses_parent_render_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["ovrtx", "remote"],
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.common import usd_cli_session
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    monkeypatch.setattr(
        usd_cli_session,
        "_parent_usd_cli_session_identity_from_environment",
        lambda: (SimpleNamespace(), tmp_path / "parent-session.json"),
    )
    monkeypatch.setattr(
        usd_cli_session,
        "_environment_remote_render_config",
        lambda: pytest.fail("attached child must reuse the parent render config"),
    )
    calls = _install_fake_usd_cli(monkeypatch)

    produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / f"attached-{backend}",
        backend=backend,
        views=("+x+y+z",),
    )

    assert calls[0]["render_config"] is None


def test_canonical_visual_ingest_rejects_stale_usd_cli_support_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(monkeypatch)
    publication = produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / "visual",
        backend="ovrtx",
    )
    Path(publication.result.artifacts[-2].path).write_text("altered", encoding="utf-8")

    with pytest.raises(VerifiedOperationError, match="stale"):
        ingest_verified_operation_result(
            publication.envelope.path,
            output_dir=tmp_path / "provided",
        )


def test_canonical_visual_leaf_rejects_output_changed_during_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(monkeypatch, mutate_during_render=output)
    with pytest.raises(VerifiedOperationError, match="changed while rendering"):
        produce_canonical_visual_evidence(
            post_mutation_usd=output,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
        )


def test_canonical_visual_leaf_rejects_when_every_view_is_blank_suspect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(
        monkeypatch,
        blank_suspect_views={"view-000", "view-001"},
    )

    with pytest.raises(
        VerifiedOperationError,
        match="every canonical OVRTX view was flagged blank_suspect",
    ):
        produce_canonical_visual_evidence(
            post_mutation_usd=output,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
            views=("+x+y+z", "+x-y+z"),
        )


def test_canonical_visual_leaf_retains_nonblank_view_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(
        monkeypatch,
        blank_suspect_views={"view-000"},
    )

    publication = produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / "visual",
        backend="ovrtx",
        views=("+x+y+z", "+x-y+z"),
    )

    assert publication.result.native_status == "pass"


def test_canonical_visual_leaf_rejects_output_flushed_during_session_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(monkeypatch, mutate_on_close=output)

    with pytest.raises(VerifiedOperationError, match="changed while rendering"):
        produce_canonical_visual_evidence(
            post_mutation_usd=output,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
        )


def test_canonical_visual_leaf_accepts_sha256_usd_cli_source_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(monkeypatch, source_revision="b" * 64)

    publication = produce_canonical_visual_evidence(
        post_mutation_usd=output,
        source_usd=source,
        output_dir=tmp_path / "visual",
        backend="ovrtx",
    )

    payload = json.loads(Path(publication.payload.path).read_text(encoding="utf-8"))
    assert payload["usd_cli_source_revision"] == "b" * 64


def test_canonical_visual_leaf_preserves_primary_error_when_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(
        monkeypatch,
        render_failure=RuntimeError("primary render failure"),
        close_failure=OSError("secondary close failure"),
    )

    with pytest.raises(VerifiedOperationError, match="primary render failure") as exc:
        produce_canonical_visual_evidence(
            post_mutation_usd=output,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
        )
    assert isinstance(exc.value.__cause__, RuntimeError)


def test_canonical_visual_leaf_surfaces_close_failure_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, dependency = _assets(tmp_path)
    from content_agent_workflows.validation import canonical_visual_evidence as visual

    monkeypatch.setattr(
        visual,
        "_usd_dependency_bindings",
        lambda _path: (execution_artifact_binding(dependency),),
    )
    _install_fake_usd_cli(
        monkeypatch,
        close_failure=OSError("close readback failure"),
    )

    with pytest.raises(VerifiedOperationError, match="could not close.*readback"):
        produce_canonical_visual_evidence(
            post_mutation_usd=output,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
        )


def test_canonical_visual_leaf_rejects_symlinked_output(tmp_path: Path) -> None:
    source, output, _dependency = _assets(tmp_path)
    link = tmp_path / "linked.usda"
    link.symlink_to(output)

    with pytest.raises(VerifiedOperationError, match="unsafe"):
        produce_canonical_visual_evidence(
            post_mutation_usd=link,
            source_usd=source,
            output_dir=tmp_path / "visual",
            backend="ovrtx",
        )
