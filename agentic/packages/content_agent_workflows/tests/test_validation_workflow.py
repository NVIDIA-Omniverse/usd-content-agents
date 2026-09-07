# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the resumable Validation Agent workflow."""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pytest
from PIL import Image
from world_understanding.utils.credentials import InlineSecretError
from world_understanding.validation import (
    ValidationEvidence,
    ValidationIssue,
    ValidationPlan,
    ValidationPlanStep,
    ValidationRenderConfig,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
)
from world_understanding.validation.scaffold_runner import (
    ScaffoldValidationStepExecutor,
)

import content_agent_workflows.common.artifacts as artifacts_module
import content_agent_workflows.validation.finalizer as finalizer_module
import content_agent_workflows.validation.workflow as workflow_module
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import (
    DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
    DomainExecutionContext,
    metadata_with_domain_execution_context,
)
from content_agent_workflows.validation import (
    FileValidationCheckpointStore,
    ValidationCancellationToken,
    ValidationCheckpointError,
    ValidationCheckpointStore,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowError,
    ValidationWorkflowIdentityMismatch,
    ValidationWorkItemState,
    recover_orphaned_validation_claims,
    request_validation_cancellation,
    run_validation_workflow,
)

TOOLBOX_PROMPT = (
    "Validate that this generated electrician's toolbox renders successfully "
    "and looks like the supplied reference image. Do not modify the asset; "
    "save a report with evidence and recommended actions."
)
AtomicJsonWriter = Callable[[str | Path, Any], Path]
AtomicJsonAtWriter = Callable[[int, str, Any], None]


def _write_source(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "Toolbox"
)
def Xform "Toolbox"
{
}
""",
        encoding="utf-8",
    )
    return path.resolve()


def _write_image(
    path: Path,
    color: tuple[int, int, int],
    *,
    image_format: str | None = None,
) -> Path:
    Image.new("RGB", (8, 8), color=color).save(path, format=image_format)
    return path.resolve()


def _request(
    source: Path,
    reference: Path,
    *,
    requested_templates: tuple[str, ...] = ("look_right", "render_valid"),
    policy: Mapping[str, Any] | None = None,
) -> ValidationRequest:
    request_policy: dict[str, Any] = {
        "visual_evidence_mode": "canonical_usd",
        "reference_image_paths": [str(reference)],
    }
    request_policy.update(policy or {})
    return ValidationRequest(
        task_description=TOOLBOX_PROMPT,
        inputs=(str(source),),
        requested_templates=requested_templates,
        render=ValidationRenderConfig(
            backend="fake",
            image_width=64,
            image_height=64,
            views=("corner",),
        ),
        policy=request_policy,
    )


class FakeValidationExecutor:
    def __init__(
        self,
        *,
        source: Path,
        token: ValidationCancellationToken | None = None,
        cancel_on_return: str | None = None,
        mutate_source_on: str | None = None,
        dependency_unavailable: bool = False,
        version_suffix: str = "v1",
    ) -> None:
        self.source = source
        self.token = token
        self.cancel_on_return = cancel_on_return
        self.mutate_source_on = mutate_source_on
        self.dependency_unavailable = dependency_unavailable
        self.calls: list[str] = []
        self.plan_calls = 0
        self.previous_results: list[tuple[str, ...]] = []
        self._template_versions = {
            "render_valid": f"fake.render-valid.{version_suffix}",
            "look_right": f"fake.look-right.{version_suffix}",
        }

    @property
    def template_versions(self) -> Mapping[str, str]:
        return self._template_versions

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        self.plan_calls += 1
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=name,
                    reason="Selected by the Wave 2 visual workflow.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Fake deterministic visual validation plan.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        self.calls.append(template_name)
        self.previous_results.append(
            tuple(result.template_name for result in context.previous_template_results)
        )
        context.working_dir.mkdir(parents=True, exist_ok=True)
        evidence = _write_image(
            context.working_dir / f"{template_name}.png",
            (230, 190, 20) if template_name == "render_valid" else (40, 40, 40),
        )
        if self.mutate_source_on == template_name:
            self.source.write_text(
                "mutated by adversarial executor\n", encoding="utf-8"
            )
        if self.cancel_on_return == template_name and self.token is not None:
            self.token.cancel()

        if self.dependency_unavailable:
            code = (
                "render.renderer_unavailable"
                if template_name == "render_valid"
                else "visual.judge_unavailable"
            )
            return ValidationTemplateResult(
                template_name=template_name,
                status="skipped",
                issues=(
                    ValidationIssue(
                        code=code,
                        severity="warn",
                        message=f"{template_name} dependency is unavailable.",
                        template_name=template_name,
                    ),
                ),
                metrics={"issue_count": 1, "vlm_invoked": False},
                metadata={
                    "runtime_render": {
                        "status": "unavailable",
                        "backend": "fake",
                        "image_paths": [],
                        "issues": [],
                        "metadata": {},
                    }
                    if template_name == "render_valid"
                    else {}
                },
            )

        metadata: dict[str, Any] = {"executor": "fake"}
        if template_name == "render_valid":
            metadata.update(
                {
                    "runtime_render": {
                        "status": "passed",
                        "backend": "fake",
                        "image_paths": [str(evidence)],
                        "render_response": None,
                        "render_output_dir": str(context.working_dir),
                        "issues": [],
                        "metadata": {},
                    },
                    "adapter_result": {
                        "status": "pass",
                        "verdict": "pass",
                        "issues": [],
                    },
                }
            )
        return ValidationTemplateResult(
            template_name=template_name,
            status="passed",
            metrics={"issue_count": 0, "vlm_invoked": template_name == "look_right"},
            evidence={"image_paths": [str(evidence)]},
            metadata=metadata,
        )


class InterruptingValidationExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, output_dir: Path) -> None:
        super().__init__(source=source)
        self.output_dir = output_dir
        self.visible_terminal_artifacts: tuple[str, ...] | None = None

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        del context
        self.calls.append(template_name)
        self.visible_terminal_artifacts = tuple(
            path.name
            for path in (
                self.output_dir / "validation_result.json",
                self.output_dir / "validation_evidence.json",
                self.output_dir / "final_summary.json",
            )
            if path.exists()
        )
        raise KeyboardInterrupt("interrupted validation execution")


class MissingRenderEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "render_valid":
            for path in result.evidence["image_paths"]:
                Path(path).unlink()
        return result


class FileUriRenderEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "render_valid":
            return result
        render_path = Path(result.evidence["image_paths"][0]).resolve()
        render_uri = f"file://localhost{render_path}"
        metadata = dict(result.metadata)
        runtime_render = dict(metadata["runtime_render"])
        runtime_render["image_paths"] = [render_uri]
        metadata["runtime_render"] = runtime_render
        return result.model_copy(
            update={
                "evidence": {"image_paths": [render_uri]},
                "metadata": metadata,
            }
        )


class RawWorkingDirectoryPathExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        raw_path = str(context.working_dir / f"{template_name}.png")
        metadata = dict(result.metadata)
        metadata["working_dir"] = str(context.working_dir)
        if template_name == "render_valid":
            runtime_render = dict(metadata["runtime_render"])
            runtime_render["image_paths"] = [raw_path]
            metadata["runtime_render"] = runtime_render
        return result.model_copy(
            update={
                "evidence": {"image_paths": [raw_path]},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=raw_path,
                        summary="Evidence returned through the pinned working path.",
                    ),
                ),
                "metadata": metadata,
            }
        )


class RawWorkingDirectoryFileUriExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        raw_uri = f"file://localhost{context.working_dir / f'{template_name}.png'}"
        return result.model_copy(
            update={
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=raw_uri,
                        summary=(
                            "Evidence returned through the pinned working "
                            "directory URI."
                        ),
                    ),
                ),
            }
        )


class PlannerOutputMutatingExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, output_path: Path) -> None:
        super().__init__(source=source)
        self.output_path = output_path

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        self.output_path.write_text("mutated by planner\n", encoding="utf-8")
        return super().plan(request, working_dir=working_dir)


class InputAliasingRenderEvidenceExecutor(FakeValidationExecutor):
    def __init__(
        self,
        *,
        source: Path,
        input_path: Path,
        hard_link: bool,
    ) -> None:
        super().__init__(source=source)
        self.input_path = input_path
        self.hard_link = hard_link

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "render_valid":
            return result
        evidence_path = self.input_path
        if self.hard_link:
            evidence_path = context.working_dir / "aliased-input.png"
            evidence_path.hardlink_to(self.input_path)
            evidence_path = evidence_path.resolve()
        metadata = dict(result.metadata)
        runtime_render = dict(metadata["runtime_render"])
        runtime_render["image_paths"] = [str(evidence_path)]
        metadata["runtime_render"] = runtime_render
        return result.model_copy(
            update={
                "evidence": {"image_paths": [str(evidence_path)]},
                "metadata": metadata,
            }
        )


class AttemptDirectorySwapExecutor(FakeValidationExecutor):
    def __init__(
        self,
        *,
        source: Path,
        output_dir: Path,
        archived_dir: Path,
        external_dir: Path,
    ) -> None:
        super().__init__(source=source)
        self.output_dir = output_dir
        self.archived_dir = archived_dir
        self.external_dir = external_dir
        self.swapped = False

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        canonical_attempt = (
            self.output_dir / "attempts" / template_name / "attempt-0001"
        )
        canonical_attempt.rename(self.archived_dir)
        self.external_dir.mkdir()
        canonical_attempt.symlink_to(self.external_dir, target_is_directory=True)
        self.swapped = True
        return super().run(template_name, context)


class NonpassingRenderExecutor(FakeValidationExecutor):
    def __init__(
        self,
        *,
        source: Path,
        status: Literal["failed", "error", "warn", "needs_refinement"],
    ) -> None:
        super().__init__(source=source)
        self.status = status

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "render_valid":
            return result
        severity: Literal["fail", "warn"] = (
            "fail" if self.status in {"failed", "error"} else "warn"
        )
        issue = ValidationIssue(
            code="render.runtime_render_failed",
            severity=severity,
            message="The renderer failed before visual comparison.",
            template_name=template_name,
        )
        return result.model_copy(
            update={
                "status": self.status,
                "issues": (issue,),
                "metrics": {**result.metrics, "issue_count": 1},
            }
        )


class MissingFailedRenderEvidenceExecutor(NonpassingRenderExecutor):
    def __init__(self, *, source: Path) -> None:
        super().__init__(source=source, status="failed")

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "render_valid":
            for path in result.evidence["image_paths"]:
                Path(path).unlink()
        return result


class MalformedEvidencePathExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "render_valid":
            return result
        return result.model_copy(
            update={
                "evidence": {"image_paths": ["malformed\0evidence.png"]},
                "metadata": {
                    **result.metadata,
                    "artifact_paths": {
                        "summary": "/tmp/malformed\0typed-evidence",
                    },
                },
            }
        )


class BlankEvidencePathExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "render_valid":
            return result
        return result.model_copy(update={"evidence": {"image_paths": [""]}})


class BlankTypedEvidencePathExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path="",
                        summary="Invalid blank typed evidence path.",
                    ),
                ),
            }
        )


class NestedRecordMetadataEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        nested_path = _write_image(
            context.working_dir / "nested-record-metadata.png",
            (30, 60, 90),
        )
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=result.evidence["image_paths"][0],
                        summary="Evidence with a structured artifact record.",
                        metadata={
                            "artifact_paths": {
                                "path": result.evidence["image_paths"][0],
                                "metadata": {
                                    "details": {
                                        "derivatives": {
                                            "front": str(nested_path),
                                        }
                                    }
                                },
                            }
                        },
                    ),
                ),
            }
        )


class UriEvidenceExecutor(FakeValidationExecutor):
    def __init__(
        self,
        *,
        source: Path,
        uri: str,
        typed: bool,
        typed_metadata: bool = False,
        scene_typed_metadata: bool = False,
        named_map_typed_metadata: bool = False,
        named_record_typed_metadata: bool = False,
        named_nested_record_typed_metadata: bool = False,
    ) -> None:
        super().__init__(source=source)
        self.uri = uri
        self.typed = typed
        self.typed_metadata = typed_metadata
        self.scene_typed_metadata = scene_typed_metadata
        self.named_map_typed_metadata = named_map_typed_metadata
        self.named_record_typed_metadata = named_record_typed_metadata
        self.named_nested_record_typed_metadata = named_nested_record_typed_metadata

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        if (
            self.typed_metadata
            or self.scene_typed_metadata
            or self.named_map_typed_metadata
            or self.named_record_typed_metadata
            or self.named_nested_record_typed_metadata
        ):
            evidence = {}
            evidence_items = (
                ValidationEvidence(
                    kind="image",
                    path=result.evidence["image_paths"][0],
                    summary="Evidence with a metadata-declared URI.",
                    metadata=(
                        {
                            "artifact_paths": {
                                "node_paths": [{"uri": self.uri}],
                            },
                        }
                        if self.scene_typed_metadata
                        else (
                            {
                                "artifact_paths": {
                                    "path": result.evidence["image_paths"][0],
                                    **(
                                        {
                                            "derivatives": {
                                                "front": self.uri,
                                            },
                                        }
                                        if self.named_nested_record_typed_metadata
                                        else {"front": self.uri}
                                    ),
                                },
                            }
                            if (
                                self.named_record_typed_metadata
                                or self.named_nested_record_typed_metadata
                            )
                            else (
                                {"images": {"front": self.uri}}
                                if self.named_map_typed_metadata
                                else {"video_path": self.uri}
                            )
                        )
                    ),
                ),
            )
        else:
            evidence = {"image_path": self.uri} if not self.typed else {}
            evidence_items = (
                (
                    ValidationEvidence(
                        kind="image",
                        path=self.uri,
                        summary="URI-backed evidence path.",
                    ),
                )
                if self.typed
                else ()
            )
        return result.model_copy(
            update={
                "evidence": evidence,
                "evidence_items": evidence_items,
                "metadata": {"executor": "fake"},
            }
        )


class SymlinkDirectoryEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        target_dir = context.working_dir / "mutable-target"
        target_dir.mkdir()
        (target_dir / "proof.txt").write_text("mutable\n", encoding="utf-8")
        evidence_dir = context.working_dir / "symlink-evidence"
        evidence_dir.mkdir()
        (evidence_dir / "linked-target").symlink_to(
            target_dir,
            target_is_directory=True,
        )
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=result.evidence["image_paths"][0],
                        summary="Evidence with an unsafe directory symlink.",
                        metadata={
                            "artifacts": {
                                "bundle": str(evidence_dir),
                            },
                        },
                    ),
                ),
            }
        )


class OrdinaryMetadataExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "metadata": {
                    **result.metadata,
                    "security": "passed",
                    "maturity": "stable",
                    "pathology": "none",
                }
            }
        )


class StructuredMetadataEvidenceExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, metadata: Mapping[str, Any]) -> None:
        super().__init__(source=source)
        self.metadata = metadata

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={"metadata": {**result.metadata, **self.metadata}}
        )


class MissingMapEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=result.evidence["image_paths"][0],
                        summary="Evidence with a missing named map artifact.",
                        metadata={"images": {"front": "never-written.webp"}},
                    ),
                ),
            }
        )


class WorkflowArtifactEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        workflow_result = Path(context.plan.artifact_paths["validation_result"])
        workflow_result.write_text("executor-owned evidence\n", encoding="utf-8")
        prior_attempt_result = (
            workflow_result.parent
            / "attempts"
            / "render_valid"
            / "attempt-0001"
            / "template_result.json"
        )
        assert prior_attempt_result.is_file()
        return result.model_copy(
            update={
                "evidence": {
                    "image_paths": [
                        *result.evidence["image_paths"],
                        str(workflow_result),
                        str(prior_attempt_result),
                    ]
                }
            }
        )


class WorkflowArtifactHardLinkEvidenceExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, evidence_path: Path) -> None:
        super().__init__(source=source)
        self.evidence_path = evidence_path

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        workflow_request = Path(context.plan.artifact_paths["validation_request"])
        self.evidence_path.hardlink_to(workflow_request)
        return result.model_copy(
            update={
                "evidence": {
                    "image_paths": [
                        *result.evidence["image_paths"],
                        str(self.evidence_path),
                    ]
                }
            }
        )


class MutatingContextExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        context.request.policy["executor_mutation"] = template_name
        context.plan.metadata["executor_mutation"] = template_name
        context.plan.steps[0].metadata["executor_mutation"] = template_name
        return super().run(template_name, context)


class WorkflowAncestorEvidenceExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, evidence_dir: Path) -> None:
        super().__init__(source=source)
        self.evidence_dir = evidence_dir

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "evidence": {
                    "image_paths": [
                        *result.evidence["image_paths"],
                        str(self.evidence_dir),
                    ]
                }
            }
        )


class MutatingReferenceExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, reference: Path) -> None:
        super().__init__(source=source)
        self.reference = reference

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "look_right":
            _write_image(self.reference, (1, 2, 3))
        return result


class MutatingRenderEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "look_right":
            render_result = context.previous_template_results[0]
            render_path = Path(render_result.evidence["image_paths"][0])
            _write_image(render_path, (3, 2, 1))
        return result


class UnsafeResultExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, secret: str) -> None:
        super().__init__(source=source)
        self.secret = secret

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        metadata = dict(result.metadata)
        metadata["api_key"] = self.secret
        return result.model_copy(update={"metadata": metadata})


class TypedEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        evidence_path = result.evidence["image_paths"][0]
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=evidence_path,
                        summary="Typed visual comparison evidence.",
                    ),
                ),
                "metadata": {"executor": "fake"},
            }
        )


class TypedMetadataEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        primary_path = result.evidence["image_paths"][0]
        supplemental_path = _write_image(
            context.working_dir / "typed-metadata.png",
            (90, 100, 110),
        )
        thumbnail_path = _write_image(
            context.working_dir / "typed-thumbnail.png",
            (120, 130, 140),
        )
        structured_image_path = _write_image(
            context.working_dir / "typed-structured-image.png",
            (150, 160, 170),
        )
        nested_detail_path = _write_image(
            context.working_dir / "typed-nested-detail.png",
            (180, 190, 200),
        )
        structured_sequence_path = _write_image(
            context.working_dir / "typed-structured-sequence.png",
            (210, 220, 230),
        )
        image_map_path = _write_image(
            context.working_dir / "typed-image-map.png",
            (230, 220, 210),
        )
        nested_sequence_path = _write_image(
            context.working_dir / "typed-nested-sequence.png",
            (200, 180, 160),
        )
        mixed_image_path = _write_image(
            context.working_dir / "typed-mixed-image.png",
            (160, 180, 200),
        )
        bare_path = _write_image(
            context.working_dir / "typed-bare-path.png",
            (140, 160, 180),
        )
        bare_paths_path = _write_image(
            context.working_dir / "typed-bare-paths.png",
            (180, 160, 140),
        )
        video_path = _write_image(
            context.working_dir / "typed-video-path.png",
            (80, 100, 120),
        )
        depth_map_path = _write_image(
            context.working_dir / "typed-depth-map-path.png",
            (120, 100, 80),
        )
        texture_path = _write_image(
            context.working_dir / "typed-texture-path.png",
            (80, 120, 100),
        )
        named_evidence_map_path = _write_image(
            context.working_dir / "typed-named-evidence-map.png",
            (100, 80, 120),
        )
        named_image_map_path = _write_image(
            context.working_dir / "typed-named-image-map.png",
            (120, 80, 100),
        )
        custom_image_map_path = _write_image(
            context.working_dir / "typed-custom-image-map.png",
            (100, 120, 80),
        )
        extensionless_image_map_path = _write_image(
            context.working_dir / "typed-custom-image-extensionless",
            (80, 100, 60),
            image_format="PNG",
        )
        extensionless_path = _write_image(
            context.working_dir / "typed-extensionless",
            (60, 80, 100),
            image_format="PNG",
        )
        generic_extensionless_path = _write_image(
            context.working_dir / "typed-generic-extensionless",
            (80, 60, 100),
            image_format="PNG",
        )
        camera_label_path = _write_image(
            context.working_dir / "typed-camera-label.png",
            (100, 60, 80),
        )
        mesh_preview_path = _write_image(
            context.working_dir / "typed-mesh-preview.png",
            (60, 100, 80),
        )
        checksum_role_path = _write_image(
            context.working_dir / "typed-checksum-role.png",
            (70, 90, 110),
        )
        nested_record_path = _write_image(
            context.working_dir / "typed-nested-record.png",
            (110, 90, 70),
        )
        nested_capture_path = _write_image(
            context.working_dir / "typed-nested-capture.png",
            (90, 70, 110),
        )
        plural_record_path = _write_image(
            context.working_dir / "typed-plural-record.png",
            (70, 110, 90),
        )
        structured_artifact_path = _write_image(
            context.working_dir / "typed-structured-artifact.png",
            (110, 70, 90),
        )
        structured_thumbnail_path = _write_image(
            context.working_dir / "typed-structured-thumbnail.png",
            (90, 110, 70),
        )
        summary_report_path = _write_image(
            context.working_dir / "typed-summary-report.png",
            (70, 110, 90),
        )
        record_diff_path = _write_image(
            context.working_dir / "typed-record-diff.png",
            (90, 70, 110),
        )
        top_plural_path = _write_image(
            context.working_dir / "typed-top-plural.png",
            (110, 90, 70),
        )
        deep_report_path = _write_image(
            context.working_dir / "typed-deep-report.png",
            (70, 90, 110),
        )
        producer_record_path = _write_image(
            context.working_dir / "typed-producer-record.png",
            (90, 110, 70),
        )
        plural_manifest_path = _write_image(
            context.working_dir / "typed-plural-manifest.png",
            (110, 70, 90),
        )
        sequence_metadata_path = _write_image(
            context.working_dir / "typed-sequence-metadata.png",
            (70, 110, 90),
        )
        contact_sheet_path = _write_image(
            context.working_dir / "typed-contact-sheet.png",
            (90, 70, 110),
        )
        overlay_path = _write_image(
            context.working_dir / "typed-overlay.png",
            (110, 90, 70),
        )
        mask_path = _write_image(
            context.working_dir / "typed-mask.png",
            (70, 90, 110),
        )
        heatmap_path = _write_image(
            context.working_dir / "typed-heatmap.png",
            (90, 110, 70),
        )
        grounding_packet_path = _write_image(
            context.working_dir / "typed-grounding-packet.png",
            (110, 70, 90),
        )
        hybrid_front_path = _write_image(
            context.working_dir / "typed-hybrid-front.png",
            (70, 110, 90),
        )
        hybrid_side_path = _write_image(
            context.working_dir / "typed-hybrid-side.png",
            (90, 70, 110),
        )
        node_preview_path = _write_image(
            context.working_dir / "typed-node-preview.png",
            (110, 90, 70),
        )
        map_camera_path = _write_image(
            context.working_dir / "typed-map-camera.png",
            (110, 70, 90),
        )
        record_camera_path = _write_image(
            context.working_dir / "typed-record-camera.png",
            (70, 110, 90),
        )
        scene_extensionless_path = _write_image(
            context.working_dir / "typed-scene-extensionless",
            (110, 90, 70),
            image_format="PNG",
        )
        sequence_camera_path = _write_image(
            context.working_dir / "typed-sequence-camera.png",
            (70, 90, 110),
        )
        root_scene_file_path = _write_image(
            context.working_dir / "typed-root-scene-file.usda",
            (90, 110, 70),
            image_format="PNG",
        )
        node_map_preview_path = _write_image(
            context.working_dir / "typed-node-map-preview.png",
            (110, 70, 90),
        )
        scene_record_uri_path = _write_image(
            context.working_dir / "typed-scene-record-uri.png",
            (70, 90, 110),
        )
        record_front_path = _write_image(
            context.working_dir / "typed-record-front.png",
            (90, 110, 70),
        )
        root_artifact_path = _write_image(
            context.working_dir / "typed-root-artifact.png",
            (110, 70, 90),
        )
        scene_record_camera_path = _write_image(
            context.working_dir / "typed-scene-record-camera.png",
            (70, 90, 110),
        )
        evidence_bundle_dir = context.working_dir / "typed-evidence-bundle"
        evidence_bundle_dir.mkdir()
        nested_derivative_path = _write_image(
            context.working_dir / "typed-nested-derivative.png",
            (90, 110, 70),
        )
        nested_metadata_artifact_path = _write_image(
            context.working_dir / "typed-nested-metadata-artifact.png",
            (110, 70, 90),
        )
        source_role_path = _write_image(
            context.working_dir / "typed-source-role.png",
            (70, 90, 110),
        )
        return result.model_copy(
            update={
                "evidence": {},
                "evidence_items": (
                    ValidationEvidence(
                        kind="manifest",
                        path=primary_path,
                        summary="Typed evidence with supplemental image artifacts.",
                        metadata={
                            "image_paths": [str(supplemental_path)],
                            "image_path_groups": {
                                "cam1": {
                                    "path": str(nested_record_path),
                                    "camera": "/World/NestedRecordCamera",
                                },
                            },
                            "artifact_paths": {
                                "checksum": "sha256:not-an-artifact-path",
                                "metadata": {
                                    "source": "renderer-artifact-map",
                                },
                                "producer": "renderer-direct",
                                "bundle": str(evidence_bundle_dir),
                                "thumbnail": str(thumbnail_path),
                                "summary": "Artifact bundle overview",
                                "preview": extensionless_path.name,
                                "camera_path": str(camera_label_path),
                                "material_path": "/World/Looks/ArtifactMetadata",
                                "light_path": "/World/Shader.png",
                                "attribute_path": "/World/Cube.size",
                                "prim_path": "/World/Cam_v1.2",
                                "shader_path": scene_extensionless_path.name,
                                "node_paths": [
                                    {
                                        "path": "/World/N",
                                        "camera_path": scene_record_camera_path.name,
                                        "preview_path": str(node_map_preview_path),
                                        "uri": (
                                            f"file://localhost"
                                            f"{scene_record_uri_path.resolve()}"
                                        ),
                                    },
                                ],
                                "views": ["front.v2", "side.v2"],
                                "report_path": str(thumbnail_path),
                                "detail": {
                                    "path": str(nested_detail_path),
                                    "caption": "Front view",
                                },
                            },
                            "path": str(bare_path),
                            "paths": [str(bare_paths_path)],
                            "video_path": str(video_path),
                            "depth_map_path": str(depth_map_path),
                            "texture_path": str(texture_path),
                            "preview_path": generic_extensionless_path.name,
                            "images": [
                                {
                                    "path": str(structured_image_path),
                                    "caption": "Reference comparison thumbnail",
                                    "camera": "/World/RecordImageCamera",
                                    "captures": {
                                        "front": str(nested_capture_path),
                                    },
                                    "image_description": "Front image description",
                                    "metadata": {
                                        "source": "renderer",
                                    },
                                    "mime_type": "image/png",
                                    "render_status": "passed",
                                    "video_fps": "24 fps",
                                    "view": "+x",
                                },
                                {"front": str(named_image_map_path)},
                            ],
                            "evidence_paths": [
                                {
                                    "path": str(structured_sequence_path),
                                    "caption": "Side view",
                                    "captions": ["Side view", "Close-up"],
                                    "camera": "/World/RecordEvidenceCamera",
                                    "media_type": "image/png",
                                    "view": "+y",
                                },
                                {
                                    "thumbnail": str(named_evidence_map_path),
                                    "caption": "Named evidence thumbnail",
                                    "kind": "image",
                                    "subject": "/World/NamedEvidenceSubject",
                                    "camera_path": "/World/Camera/NamedEvidence",
                                },
                                {
                                    "paths": [str(plural_record_path)],
                                    "view": "+z",
                                },
                            ],
                            "report_paths": {
                                "details": [[str(nested_sequence_path)]],
                                "producer": "validator",
                                "summary": "QA report overview",
                                "summary_report": str(summary_report_path),
                            },
                            "manifest_paths": {
                                "paths": [str(plural_manifest_path)],
                                "summary": "QA manifest overview",
                            },
                            "hybrid_image_paths": {
                                "front": str(hybrid_front_path),
                                "paths": [str(hybrid_side_path)],
                                "producer": "hybrid-renderer",
                                "summary": "Hybrid views",
                            },
                            "preview_image_paths": {
                                "front": {
                                    "path": str(producer_record_path),
                                    "producer": "renderer",
                                    "metadata": [
                                        {
                                            "preview_path": str(sequence_metadata_path),
                                            "producer": "renderer",
                                        },
                                    ],
                                },
                            },
                            "default_root_path": "/World",
                            "mesh_path": "/World/Body/Mesh",
                            "mesh_preview_path": str(mesh_preview_path),
                            "physics_scene_paths": ["/World/PhysicsScene"],
                            "collection_path": "/World/Collection",
                            "children_paths": ["/World/Body/Child"],
                            "related_paths": ["/World/Body", "/World/Body/Mesh"],
                            "material_path": "/World/Looks/Paint",
                            "primvar_path": "/World/Mesh.primvars:st",
                            "output_render_mesh_path": "/World/RenderMesh",
                            "parent_path": "/World/Looks",
                            "body_root_path": "/World",
                            "source_body_path": "/World/SourceBody",
                            "camera_path": "/World/Camera",
                            "render_camera_paths": [
                                "/World/Cameras/TextureAgentFinal",
                            ],
                            "shader_path": "/World/Looks/Paint/Shader",
                            "node_paths": ["/World/Looks/Paint/Shader/Node"],
                            "joint_path": "/World/Joints/Hinge",
                            "skeleton_path": "/World/Skeleton",
                            "xform_path": "/World/Body",
                            "collision_paths": ["/Root/Body/Collision"],
                            "mass_authoring_path": "/Asset/Body",
                            "selection_paths": ["/asset/body", "/root/body"],
                            "subset_path": "/World/Mesh/face_0",
                            "collider_paths": ["/asset/body/collider"],
                            "attribute_path": "/World/Body.visibility",
                            "relationship_target_path": "/World/Target",
                            "prim_paths": {
                                "selected": ["/World/Body"],
                            },
                        },
                    ),
                    ValidationEvidence(
                        kind="image",
                        path=primary_path,
                        summary="Typed evidence with a named image map.",
                        metadata={
                            "images": {
                                "front": str(image_map_path),
                                "front_path": str(named_image_map_path),
                                "camera_01": str(custom_image_map_path),
                                "camera_02": extensionless_image_map_path.name,
                                "paths": [str(mixed_image_path)],
                                "captions": ["Front map view"],
                                "content_type": "image/png",
                                "color_space": "sRGB",
                                "coordinate_system": "right-handed",
                                "duration": "2s",
                                "height": 8,
                                "metadata": {
                                    "image_path": str(structured_thumbnail_path),
                                    "source": "renderer-map",
                                },
                                "view_reports": {
                                    "report_paths": {
                                        "main": str(deep_report_path),
                                        "caption": "Deep report overview",
                                    },
                                },
                                "views": ["front", "side"],
                                "model": "qwen2.5-vl",
                                "provider": "nvidia.nim",
                                "revision": "front.v2",
                                "mode": "RGB",
                                "render_time": "0.3s",
                                "sha256": "sha256:not-an-artifact-path",
                                "id": "typed images id.png",
                                "summary": "Front views",
                                "width": 8,
                            },
                        },
                    ),
                    ValidationEvidence(
                        kind="report",
                        path=primary_path,
                        summary="Typed evidence with a structured artifact record.",
                        metadata={
                            "artifact_paths": {
                                "path": str(structured_artifact_path),
                                "front": str(record_front_path),
                                "derivatives": {
                                    "front": str(nested_derivative_path),
                                },
                                "metadata": [
                                    {
                                        "command": "render completed",
                                        "artifacts": {
                                            "preview": str(
                                                nested_metadata_artifact_path
                                            ),
                                        },
                                    },
                                ],
                                "source": str(source_role_path),
                                "message": "render completed",
                                "reason": "reference comparison passed",
                                "command": "render finished",
                                "summary": "rendered to output.png",
                                "caption": "Structured artifact record",
                                "contact_sheet": str(contact_sheet_path),
                                "diff": str(record_diff_path),
                                "heatmap": str(heatmap_path),
                                "mask": str(mask_path),
                                "overlay": str(overlay_path),
                                "thumbnail": str(structured_thumbnail_path),
                                "visual_grounding_packet": str(grounding_packet_path),
                                "id": "run.1",
                                "camera_path": str(record_camera_path),
                            },
                        },
                    ),
                    ValidationEvidence(
                        kind="report",
                        path=primary_path,
                        summary="Typed evidence with a plural artifact record.",
                        metadata={
                            "artifact_paths": {
                                "paths": [str(top_plural_path)],
                                "caption": "Plural artifact record",
                            },
                        },
                    ),
                    ValidationEvidence(
                        kind="report",
                        path=primary_path,
                        summary="Typed evidence with plural named artifact roles.",
                        metadata={
                            "artifact_paths": {
                                "report_paths": [str(summary_report_path)],
                                "file_paths": [str(checksum_role_path)],
                                "manifest_paths": [str(plural_manifest_path)],
                                "evidence_paths": [str(supplemental_path)],
                                "summary": "Plural artifact collection",
                                "camera_path": str(map_camera_path),
                                "camera_paths": [str(sequence_camera_path)],
                            },
                        },
                    ),
                    ValidationEvidence(
                        kind="report",
                        path=primary_path,
                        summary="Typed evidence with inline authoritative metadata.",
                        metadata={
                            "artifact_paths": {
                                "thumbnail": str(thumbnail_path),
                                "id": "run.1",
                                "checksum": "sha256:inline-metadata",
                                "summary": "Rendered front, side. QA pass",
                            },
                        },
                    ),
                ),
                "metadata": {
                    "executor": "fake",
                    "camera_path": "../missing-shot.usd",
                    "root_path": "/",
                    "node_paths": [
                        "/World/ResultShader/Node",
                        {
                            "path": "/World/Looks/StructuredShader",
                            "preview_path": str(node_preview_path),
                        },
                    ],
                    "shader_path": "/World/ResultShader",
                    "light_path": root_scene_file_path.name,
                    "artifacts": {"front": str(root_artifact_path)},
                    "visual_grounding_packet": str(grounding_packet_path),
                },
            }
        )


class MissingTypedEvidenceExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        return result.model_copy(
            update={
                "evidence_items": (
                    ValidationEvidence(
                        kind="image",
                        path=str(context.working_dir / "missing-declared.png"),
                        summary="Evidence that was declared but never written.",
                    ),
                ),
            }
        )


class RelativeRenderHandoffExecutor:
    def __init__(self, *, config_base_dir: Path) -> None:
        self.delegate = ScaffoldValidationStepExecutor(config_base_dir)

    @property
    def template_versions(self) -> Mapping[str, str]:
        return self.delegate.template_versions

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        return self.delegate.plan(request, working_dir=working_dir)

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        if template_name != "render_valid":
            return self.delegate.run(template_name, context)
        _write_image(context.working_dir / "render.png", (220, 180, 0))
        return ValidationTemplateResult(
            template_name="render_valid",
            status="passed",
            evidence={"image_paths": ["render.png"]},
            metadata={
                "runtime_render": {
                    "status": "completed",
                    "backend": "fake",
                    "image_paths": ["render.png"],
                    "render_response": {
                        "status": "completed",
                        "results": [
                            {
                                "camera": "corner",
                                "camera_path": "/ValidationAgentCameras/corner",
                                "images": ["render.png"],
                                "status": "success",
                            }
                        ],
                    },
                    "render_output_dir": ".",
                    "issues": [],
                    "metadata": {},
                },
                "adapter_result": {
                    "status": "pass",
                    "verdict": "pass",
                    "issues": [],
                },
            },
        )


class SourceReadFailureExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path) -> None:
        super().__init__(source=source)
        self.block_source_reads = False

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        self.block_source_reads = True
        return result


class PersistedCancellationExecutor(FakeValidationExecutor):
    def __init__(self, *, source: Path, output_dir: Path) -> None:
        super().__init__(source=source)
        self.output_dir = output_dir

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name == "look_right":
            request_validation_cancellation(
                self.output_dir,
                reason="Stop the in-flight visual comparison.",
            )
        return result


class MismatchedTemplateExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        return result.model_copy(
            update={
                "template_name": (
                    "look_right" if template_name == "render_valid" else "render_valid"
                )
            }
        )


class ReservedIntegrityWarningExecutor(FakeValidationExecutor):
    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        result = super().run(template_name, context)
        if template_name != "look_right":
            return result
        issues = (
            ValidationIssue(
                code="asset.expected_defect",
                severity="fail",
                message="The fixture contains its expected asset defect.",
                template_name=template_name,
            ),
            ValidationIssue(
                code="validation.accepted_evidence_stale",
                severity="warn",
                message="Executor-supplied warning using a reserved integrity code.",
                template_name=template_name,
            ),
        )
        return result.model_copy(
            update={
                "status": "failed",
                "issues": issues,
                "metrics": {**result.metrics, "issue_count": len(issues)},
            }
        )


class CancelBeforeLookClaimStore(FileValidationCheckpointStore):
    def __init__(self, path: Path, *, output_dir: Path) -> None:
        super().__init__(path)
        self.output_dir = output_dir
        self.triggered = False

    def update(self, mutation: Any) -> Any:
        current = self.load()
        if (
            not self.triggered
            and current is not None
            and current.records[0].accepted_result is not None
            and current.records[1].accepted_result is None
            and not current.cancellation_requested
        ):
            self.triggered = True
            request_validation_cancellation(
                self.output_dir,
                reason="Cancel immediately before the next atomic claim.",
            )
        return super().update(mutation)


class CancelAfterTerminalLoadStore(FileValidationCheckpointStore):
    def __init__(self, path: Path, *, output_dir: Path) -> None:
        super().__init__(path)
        self.output_dir = output_dir
        self.triggered = False

    def finalize(self, finalization: Any) -> Any:
        current = super().load()
        if (
            not self.triggered
            and current is not None
            and all(record.accepted_result is not None for record in current.records)
        ):
            self.triggered = True
            request_validation_cancellation(
                self.output_dir,
                reason="Cancel after the final accepted result.",
                checkpoint_store=FileValidationCheckpointStore(self.path),
            )
        return super().finalize(finalization)


class LegacyValidationCheckpointStore(ValidationCheckpointStore):
    def __init__(self, path: Path) -> None:
        self.delegate = FileValidationCheckpointStore(path)

    @property
    def path(self) -> Path:
        return self.delegate.path

    def load(self) -> ValidationWorkflowCheckpoint | None:
        return self.delegate.load()

    def create(
        self,
        checkpoint: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        return self.delegate.create(checkpoint)

    def update(self, mutation: Any) -> ValidationWorkflowCheckpoint:
        return self.delegate.update(mutation)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = _write_source(tmp_path / "toolbox.usda")
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    return source, reference


def test_toolbox_workflow_binds_order_and_writes_stable_artifacts(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    source_digest = file_sha256(source)
    executor = FakeValidationExecutor(source=source)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid", "look_right"]
    assert executor.previous_results == [(), ("render_valid",)]
    assert tuple(step.template_name for step in run.plan.steps) == (
        "render_valid",
        "look_right",
    )
    render_binding = run.plan.steps[0].metadata["agentic_work_item"]
    look_binding = run.plan.steps[1].metadata["agentic_work_item"]
    assert render_binding["depends_on"] == []
    assert look_binding["depends_on"] == ["validation:render_valid"]
    for binding in (render_binding, look_binding):
        assert len(binding["identity_digest"]) == 64
        assert len(binding["request_digest"]) == 64
        assert len(binding["policy_digest"]) == 64
        assert len(binding["backend_digest"]) == 64
        assert binding["source_artifacts"][0]["sha256"] == source_digest
        assert binding["reference_artifacts"][0]["sha256"] == file_sha256(reference)
    assert run.request.requested_templates == ("render_valid", "look_right")
    assert run.result.verdict == "pass"
    assert "no validation action is required" in (run.result.recommended_action or "")
    assert file_sha256(source) == source_digest
    assert all(
        Path(path).is_file()
        for path in (
            run.request_path,
            run.plan_path,
            run.result_path,
            run.checkpoint_path,
            run.evidence_path,
            run.final_summary_path,
        )
    )
    evidence = json.loads(Path(run.evidence_path).read_text(encoding="utf-8"))
    assert evidence["source_unchanged"] is True
    assert set(evidence["templates"]) == {"render_valid", "look_right"}


def test_standalone_context_does_not_claim_embedded_completion(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    context = DomainExecutionContext(
        schema_version=DOMAIN_EXECUTION_CONTEXT_SCHEMA_VERSION_V2,
        domain="validation",
        mode="standalone",
        reasoning_loop_owner="domain_child_agent",
    )
    request = _request(source, reference).model_copy(
        update={
            "metadata": metadata_with_domain_execution_context({}, context),
        }
    )

    run = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert "embedded_execution" not in run.result.metadata
    assert "native_verdict_role" not in run.result.metadata
    assert "look_right_role" not in run.result.metadata
    assert "semantic_completion_authority" not in run.result.metadata


def test_cancel_after_render_then_resume_executes_only_look_right(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    token = ValidationCancellationToken()
    first_executor = FakeValidationExecutor(source=source)
    cancelled_after_render = False

    def cancel_after_render(checkpoint: object) -> None:
        nonlocal cancelled_after_render
        if cancelled_after_render:
            return
        records = getattr(checkpoint, "records")
        if records[0].accepted_result is not None:
            cancelled_after_render = True
            token.cancel()

    first = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=first_executor,
        cancellation_token=token,
        progress_callback=cancel_after_render,
    )

    assert first.status.value == "cancelled"
    assert first_executor.calls == ["render_valid"]
    assert first.checkpoint.records[0].accepted_result is not None
    assert first.checkpoint.records[1].accepted_result is None

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resume_executor.previous_results == [("render_valid",)]
    assert resumed.status.value == "completed"
    assert resumed.result.verdict == "pass"


def test_resume_reopened_work_invalidates_prior_terminal_bundle_before_execution(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"
    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    render_accepted = first.checkpoint.records[0].accepted_result
    assert render_accepted is not None
    render_evidence_path = Path(render_accepted.result.evidence["image_paths"][0])
    render_evidence_path.unlink()

    resume_executor = InterruptingValidationExecutor(
        source=source,
        output_dir=output_dir,
    )
    with pytest.raises(KeyboardInterrupt, match="interrupted validation execution"):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=resume_executor,
            resume=True,
        )

    assert resume_executor.calls == ["render_valid"]
    assert resume_executor.visible_terminal_artifacts == ()
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_fresh_run_without_checkpoint_invalidates_prior_terminal_bundle(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"
    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    Path(first.checkpoint_path).unlink()
    shutil.rmtree(output_dir / "attempts")

    executor = InterruptingValidationExecutor(
        source=source,
        output_dir=output_dir,
    )
    with pytest.raises(KeyboardInterrupt, match="interrupted validation execution"):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert executor.calls == ["render_valid"]
    assert executor.visible_terminal_artifacts == ()
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_unchanged_completed_resume_preserves_terminal_bundle_and_revision(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    assert first.result.verdict == "pass"
    published_result = (output_dir / "validation_result.json").read_text(
        encoding="utf-8"
    )
    published_revision = first.result.metadata["checkpoint_revision"]

    def abort_first_report(checkpoint: ValidationWorkflowCheckpoint) -> None:
        raise RuntimeError("injected progress callback failure")

    resume_executor = FakeValidationExecutor(source=source)
    with pytest.raises(RuntimeError, match="injected progress callback failure"):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=resume_executor,
            resume=True,
            progress_callback=abort_first_report,
        )

    assert resume_executor.calls == []
    reloaded = FileValidationCheckpointStore(
        output_dir / "validation_checkpoint.json"
    ).load()
    assert reloaded is not None
    assert reloaded.revision == published_revision
    assert (output_dir / "validation_result.json").read_text(
        encoding="utf-8"
    ) == published_result
    assert (output_dir / "validation_evidence.json").exists()
    assert (output_dir / "final_summary.json").exists()


def test_mutating_completed_resume_drops_terminal_bundle_before_revision_bump(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    assert first.result.verdict == "pass"
    published_revision = first.result.metadata["checkpoint_revision"]

    # A foreign or legacy writer can leave a stale error beside an accepted
    # result. Resume repairs that record without reopening any work, so every
    # record still holds an accepted result while the checkpoint changes.
    checkpoint_path = output_dir / "validation_checkpoint.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert all(
        record["accepted_result"] is not None
        for record in checkpoint_payload["records"]
    )
    checkpoint_payload["records"][-1]["last_error"] = "stale error from a prior writer"
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def abort_first_report(checkpoint: ValidationWorkflowCheckpoint) -> None:
        raise RuntimeError("injected progress callback failure")

    resume_executor = FakeValidationExecutor(source=source)
    with pytest.raises(RuntimeError, match="injected progress callback failure"):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=resume_executor,
            resume=True,
            progress_callback=abort_first_report,
        )

    assert resume_executor.calls == []
    reloaded = FileValidationCheckpointStore(checkpoint_path).load()
    assert reloaded is not None
    assert reloaded.revision > published_revision
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()

    republished = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        resume=True,
    )
    assert republished.result.verdict == "pass"
    assert republished.result.metadata["checkpoint_revision"] == reloaded.revision


def test_progress_callback_failure_releases_unstarted_claim(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    def abort_on_running(checkpoint: ValidationWorkflowCheckpoint) -> None:
        if any(
            record.state == ValidationWorkItemState.RUNNING
            for record in checkpoint.records
        ):
            raise RuntimeError("injected progress callback failure")

    with pytest.raises(RuntimeError, match="injected progress callback failure"):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            progress_callback=abort_on_running,
        )

    checkpoint = FileValidationCheckpointStore(
        output_dir / "validation_checkpoint.json"
    ).load()
    assert checkpoint is not None
    assert checkpoint.records[0].state == ValidationWorkItemState.PENDING
    assert checkpoint.records[0].accepted_result is None

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )
    assert resume_executor.calls == ["render_valid", "look_right"]
    assert resumed.result.verdict == "pass"


def test_cancellation_wins_over_late_success_and_can_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    token = ValidationCancellationToken()
    executor = FakeValidationExecutor(
        source=source,
        token=token,
        cancel_on_return="look_right",
    )

    cancelled = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
        cancellation_token=token,
    )

    assert cancelled.status.value == "cancelled"
    assert cancelled.checkpoint.records[0].accepted_result is not None
    assert cancelled.checkpoint.records[1].state.value == "cancelled"
    assert cancelled.checkpoint.records[1].accepted_result is None

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )
    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_persisted_cancellation_wins_while_look_right_is_running(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    executor = PersistedCancellationExecutor(
        source=source,
        output_dir=output_dir,
    )

    cancelled = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid", "look_right"]
    assert cancelled.status.value == "cancelled"
    assert cancelled.checkpoint.records[0].accepted_result is not None
    assert cancelled.checkpoint.records[1].state.value == "cancelled"
    assert cancelled.checkpoint.records[1].accepted_result is None


def test_atomic_claim_does_not_start_work_after_persisted_cancellation(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    executor = FakeValidationExecutor(source=source)
    store = CancelBeforeLookClaimStore(
        output_dir / "validation_checkpoint.json",
        output_dir=output_dir,
    )

    cancelled = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
        checkpoint_store=store,
    )

    assert store.triggered is True
    assert executor.calls == ["render_valid"]
    assert cancelled.status.value == "cancelled"
    assert cancelled.checkpoint.records[1].accepted_result is None


def test_cancellation_after_all_results_are_accepted_is_a_noop(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    store = CancelAfterTerminalLoadStore(
        output_dir / "validation_checkpoint.json",
        output_dir=output_dir,
    )

    completed = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=store,
    )

    assert store.triggered is True
    assert completed.status.value == "completed"
    assert completed.checkpoint.cancellation_requested is False
    persisted = store.load()
    assert persisted is not None
    assert persisted.cancellation_requested is False
    summary = json.loads(Path(completed.final_summary_path).read_text(encoding="utf-8"))
    assert summary["status"] == "completed"


def test_unfinished_record_after_final_commit_cannot_publish_completion(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    store = FileValidationCheckpointStore(output_dir / "validation_checkpoint.json")
    reopened_terminal_record = False

    def reopen_look_right_after_completion(checkpoint: object) -> None:
        nonlocal reopened_terminal_record
        records = getattr(checkpoint, "records")
        if reopened_terminal_record or not all(
            record.accepted_result is not None for record in records
        ):
            return
        reopened_terminal_record = True

        def reopen(
            current: ValidationWorkflowCheckpoint,
        ) -> ValidationWorkflowCheckpoint:
            current_records = list(current.records)
            current_records[1] = current_records[1].model_copy(
                update={
                    "state": ValidationWorkItemState.PENDING,
                    "accepted_result": None,
                    "finished_at": None,
                }
            )
            return current.model_copy(update={"records": tuple(current_records)})

        store.update(reopen)

    with pytest.raises(
        ValidationCheckpointError,
        match="unfinished work and cannot be finalized",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
            progress_callback=reopen_look_right_after_completion,
        )

    assert reopened_terminal_record is True
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        checkpoint_store=store,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_checkpoint_finalization_excludes_concurrent_mutation(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    store = FileValidationCheckpointStore(output_dir / "validation_checkpoint.json")
    run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=store,
    )
    mutation_started = threading.Event()
    mutation_finished = threading.Event()
    competing_store = FileValidationCheckpointStore(store.path)

    def mutate() -> None:
        mutation_started.set()
        competing_store.update(lambda current: current)
        mutation_finished.set()

    mutation_thread = threading.Thread(target=mutate)

    def verify_locked(checkpoint: ValidationWorkflowCheckpoint) -> int:
        mutation_thread.start()
        assert mutation_started.wait(timeout=1)
        assert mutation_finished.wait(timeout=0.05) is False
        return checkpoint.revision

    revision = store.finalize(verify_locked)
    mutation_thread.join(timeout=1)

    assert mutation_finished.is_set()
    persisted = store.load()
    assert persisted is not None
    assert revision == persisted.revision


def test_checkpoint_store_without_atomic_finalization_fails_before_execution(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    store = LegacyValidationCheckpointStore(output_dir / "validation_checkpoint.json")

    executor = FakeValidationExecutor(source=source)
    assert (
        getattr(store.finalize, "__func__", None) is ValidationCheckpointStore.finalize
    )
    with pytest.raises(
        ValidationWorkflowError,
        match="must implement atomic finalize",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
            checkpoint_store=store,
        )

    assert executor.plan_calls == 0
    assert executor.calls == []
    assert not output_dir.exists()


def test_concurrent_resume_rejects_active_claim_until_cancelled(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    token = ValidationCancellationToken()

    def cancel_after_render(checkpoint: object) -> None:
        records = getattr(checkpoint, "records")
        if records[0].accepted_result is not None:
            token.cancel()

    run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        cancellation_token=token,
        progress_callback=cancel_after_render,
    )
    store = FileValidationCheckpointStore(output_dir / "validation_checkpoint.json")

    def mark_look_right_active(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        records = list(current.records)
        records[1] = records[1].model_copy(
            update={
                "state": ValidationWorkItemState.RUNNING,
                "started_at": datetime.now(UTC),
                "finished_at": None,
            }
        )
        return current.model_copy(
            update={
                "records": tuple(records),
                "cancellation_requested": False,
                "cancellation_reason": None,
            }
        )

    active = store.update(mark_look_right_active)
    active_revision = active.revision

    with pytest.raises(
        ValidationCheckpointError,
        match="another runner has active validation work",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            resume=True,
        )

    persisted = store.load()
    assert persisted is not None
    assert persisted.revision == active_revision
    assert persisted.records[1].state == ValidationWorkItemState.RUNNING

    recover_orphaned_validation_claims(
        output_dir,
        reason="Recover the orphaned active claim.",
    )
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.status.value == "completed"
    assert resumed.result.verdict == "pass"


def test_persisted_cancellation_keeps_live_attempt_fenced_until_acknowledged(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    entered = threading.Event()
    release = threading.Event()
    first_runs: list[Any] = []
    first_errors: list[BaseException] = []

    class BlockingLookRightExecutor(FakeValidationExecutor):
        def run(
            self,
            template_name: str,
            context: ValidationTemplateContext,
        ) -> ValidationTemplateResult:
            result = super().run(template_name, context)
            if template_name == "look_right":
                entered.set()
                if not release.wait(timeout=60):
                    raise TimeoutError("Test did not release look_right")
            return result

    def run_first() -> None:
        try:
            first_runs.append(
                run_validation_workflow(
                    _request(source, reference),
                    output_dir=output_dir,
                    config_base_dir=tmp_path,
                    executor=BlockingLookRightExecutor(source=source),
                )
            )
        except BaseException as exc:
            first_errors.append(exc)

    first_thread = threading.Thread(target=run_first)
    first_thread.start()
    try:
        assert entered.wait(timeout=60)
        cancelled = request_validation_cancellation(
            output_dir,
            reason="Stop the live visual comparison.",
        )
        live_record = cancelled.records[1]
        assert cancelled.cancellation_requested is True
        assert live_record.state == ValidationWorkItemState.RUNNING
        assert live_record.finished_at is None
        second_executor = FakeValidationExecutor(source=source)

        with pytest.raises(
            ValidationCheckpointError,
            match="another runner has active validation work",
        ):
            run_validation_workflow(
                _request(source, reference),
                output_dir=output_dir,
                config_base_dir=tmp_path,
                executor=second_executor,
                resume=True,
            )

        assert second_executor.calls == []
    finally:
        release.set()
        first_thread.join(timeout=60)
    assert not first_thread.is_alive()
    assert first_errors == []
    assert len(first_runs) == 1
    assert first_runs[0].status.value == "cancelled"
    assert first_runs[0].checkpoint.records[1].state.value == "cancelled"

    resumed_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resumed_executor,
        resume=True,
    )
    assert resumed_executor.calls == ["look_right"]
    assert resumed.status.value == "completed"


def test_mismatched_executor_template_name_is_rejected_before_checkpointing(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=MismatchedTemplateExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    assert run.checkpoint.records[0].accepted_result is not None
    accepted = run.checkpoint.records[0].accepted_result
    assert accepted.result.template_name == "render_valid"
    assert accepted.result.status == "error"
    assert accepted.result.issues[0].code == "validation.template_result_mismatch"
    assert run.checkpoint.records[1].accepted_result is not None
    assert (
        run.checkpoint.records[1].accepted_result.result.template_name == "look_right"
    )
    assert FileValidationCheckpointStore(run.checkpoint_path).load() == run.checkpoint


@pytest.mark.parametrize(
    "drift",
    ["prompt", "source", "reference", "policy", "backend", "template_version"],
)
def test_resume_rejects_identity_drift(tmp_path: Path, drift: str) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    executor = FakeValidationExecutor(
        source=source,
        version_suffix="v2" if drift == "template_version" else "v1",
    )
    changed = request
    if drift == "prompt":
        changed = request.model_copy(
            update={"task_description": f"{TOOLBOX_PROMPT} Check the handle."}
        )
    elif drift == "source":
        source.write_text("changed source\n", encoding="utf-8")
    elif drift == "reference":
        _write_image(reference, (0, 0, 0))
    elif drift == "policy":
        changed = request.model_copy(
            update={"policy": {**request.policy, "look_right_pass_threshold": 0.9}}
        )
    elif drift == "backend":
        changed = request.model_copy(
            update={
                "render": request.render.model_copy(
                    update={"backend": "different-fake"}
                )
            }
        )

    with pytest.raises(ValidationWorkflowIdentityMismatch):
        run_validation_workflow(
            changed,
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=executor,
            resume=True,
        )


def test_resume_rejects_tampered_identity_envelope(tmp_path: Path) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(run.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["workflow_identity"]["request_digest"] = "0" * 64
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    executor = FakeValidationExecutor(source=source)

    with pytest.raises(
        ValidationCheckpointError,
        match="identity_digest must authenticate the complete workflow identity",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
            resume=True,
        )

    assert executor.plan_calls == 0
    assert executor.calls == []


def test_missing_source_is_rejected_before_planning_or_checkpointing(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "missing-toolbox.usda"
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    executor = FakeValidationExecutor(source=missing_source)
    output_dir = tmp_path / "run"

    with pytest.raises(
        ValidationWorkflowError,
        match="source input artifacts do not exist",
    ):
        run_validation_workflow(
            _request(missing_source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.calls == []
    assert not output_dir.exists()


def test_resume_rejects_usd_dependency_drift(tmp_path: Path) -> None:
    dependency = tmp_path / "appearance.usda"
    dependency.write_text("#usda 1.0\n# yellow appearance\n", encoding="utf-8")
    source = tmp_path / "toolbox.usda"
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [ @appearance.usda@ ]\n)\n",
        encoding="utf-8",
    )
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    request = _request(source, reference)

    first = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert dependency.resolve() in {
        Path(artifact.path)
        for artifact in first.checkpoint.workflow_identity.source_artifacts
    }
    dependency.write_text(
        "#usda 1.0\n# changed blue appearance\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationWorkflowIdentityMismatch):
        run_validation_workflow(
            request,
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            resume=True,
        )


def test_resume_rejects_usdz_package_member_drift(tmp_path: Path) -> None:
    from pxr import UsdUtils

    package_layer = tmp_path / "package_layer.usda"
    package_layer.write_text("#usda 1.0\n# yellow package\n", encoding="utf-8")
    package = tmp_path / "library.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(package_layer), str(package))
    source = tmp_path / "toolbox.usda"
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [ @library.usdz[package_layer.usda]@ ]\n)\n",
        encoding="utf-8",
    )
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    request = _request(source, reference)

    first = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert package.resolve() in {
        Path(artifact.path)
        for artifact in first.checkpoint.workflow_identity.source_artifacts
    }
    package_layer.write_text("#usda 1.0\n# blue package\n", encoding="utf-8")
    replacement = tmp_path / "replacement.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(package_layer), str(replacement))
    replacement.replace(package)

    with pytest.raises(ValidationWorkflowIdentityMismatch):
        run_validation_workflow(
            request,
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            resume=True,
        )


def test_resume_rejects_checkpoint_result_that_disagrees_with_attempt_artifact(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    first = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=ReservedIntegrityWarningExecutor(source=source),
    )
    assert first.result.verdict == "fail"

    checkpoint_path = Path(first.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    embedded = payload["records"][1]["accepted_result"]["result"]
    embedded["status"] = "passed"
    embedded["issues"] = []
    embedded["metrics"]["issue_count"] = 0
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"
    persisted = json.loads(
        Path(resumed.checkpoint.records[1].accepted_result.result_path).read_text(
            encoding="utf-8"
        )
    )
    assert persisted["status"] == "passed"


def test_stale_render_evidence_is_regenerated_before_look_right(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    token = ValidationCancellationToken()
    first_executor = FakeValidationExecutor(source=source)

    def cancel_after_render(checkpoint: object) -> None:
        records = getattr(checkpoint, "records")
        if records[0].accepted_result is not None:
            token.cancel()

    first = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=first_executor,
        cancellation_token=token,
        progress_callback=cancel_after_render,
    )
    render_artifact = first.checkpoint.records[0].accepted_result.evidence_artifacts[0]
    Path(render_artifact.path).unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["render_valid", "look_right"]
    assert resumed.result.verdict == "pass"


def test_render_evidence_accepts_local_file_uri(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FileUriRenderEvidenceExecutor(source=source),
    )

    assert run.result.verdict == "pass"
    accepted = run.checkpoint.records[0].accepted_result
    assert accepted is not None
    render_paths = {
        artifact.path
        for artifact in accepted.evidence_artifacts
        if artifact.kind == "file"
    }
    assert any(path.endswith("/render_valid.png") for path in render_paths)


def test_stale_typed_look_right_evidence_is_rerun_on_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    first_executor = TypedEvidenceExecutor(source=source)
    first = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=first_executor,
    )
    typed_evidence = first.checkpoint.records[1].accepted_result.evidence_artifacts
    assert len(typed_evidence) == 1
    Path(typed_evidence[0].path).unlink()

    resume_executor = TypedEvidenceExecutor(source=source)
    resumed = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_checkpoint_cannot_drop_declared_evidence_and_reuse_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(first.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["records"][1]["accepted_result"]["evidence_artifacts"] = []
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"
    accepted = resumed.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert accepted.evidence_artifacts


def test_checkpoint_cannot_reuse_result_outside_exact_attempt_path(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(first.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    accepted_payload = payload["records"][1]["accepted_result"]
    original_result_path = Path(accepted_payload["result_path"])
    external_result_path = tmp_path / "external-template-result.json"
    external_result_path.write_bytes(original_result_path.read_bytes())
    accepted_payload["result_path"] = str(external_result_path)
    accepted_payload["result_sha256"] = file_sha256(external_result_path)
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_checkpoint_cannot_reuse_symlinked_result_at_exact_attempt_path(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    result_path = Path(accepted.result_path)
    external_result = tmp_path / "external-template-result.json"
    external_result.write_bytes(result_path.read_bytes())
    external_before = external_result.read_bytes()
    result_path.unlink()
    result_path.symlink_to(external_result)

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"
    assert external_result.read_bytes() == external_before


def test_attempt_directory_swap_cannot_redirect_executor_writes(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    archived_dir = tmp_path / "archived-attempt"
    external_dir = tmp_path / "external-attempt"
    executor = AttemptDirectorySwapExecutor(
        source=source,
        output_dir=output_dir,
        archived_dir=archived_dir,
        external_dir=external_dir,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    canonical_attempt = output_dir / "attempts" / "render_valid" / "attempt-0001"
    assert executor.swapped is True
    assert run.status.value == "cancelled"
    assert run.result.verdict != "pass"
    assert canonical_attempt.is_symlink()
    assert not any(external_dir.iterdir())
    assert (archived_dir / "render_valid.png").is_file()
    assert all(
        record.state != ValidationWorkItemState.RUNNING
        for record in run.checkpoint.records
    )


def test_preexisting_attempt_directory_cannot_redirect_executor_writes(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    attempt_dir = output_dir / "attempts" / "render_valid" / "attempt-0001"
    attempt_dir.mkdir(parents=True)
    external_dir = tmp_path / "external-render-output"
    external_dir.mkdir()
    sentinel = external_dir / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    (attempt_dir / "renders").symlink_to(
        external_dir,
        target_is_directory=True,
    )
    executor = FakeValidationExecutor(source=source)

    with pytest.raises(
        ValidationWorkflowError,
        match="attempt directory already exists",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert executor.calls == []
    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert (attempt_dir / "renders").is_symlink()


def test_attempt_execution_fails_closed_without_descriptor_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(source=source)
    original_is_dir = Path.is_dir
    descriptor_prefix = "/proc/self/fd/"

    def hide_descriptor_directory(path: Path) -> bool:
        if str(path).startswith(descriptor_prefix):
            return False
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", hide_descriptor_directory)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == []
    assert run.result.verdict == "fail"
    assert "validation.template_execution_error" in {
        issue.code for issue in run.result.issues
    }


def test_attempt_execution_uses_proc_self_across_pid_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(source=source)
    monkeypatch.setattr(workflow_module.os, "getpid", lambda: 2_147_483_647)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid", "look_right"]
    assert run.result.verdict == "pass"


def test_scaffold_attempt_swap_cannot_redirect_renderer_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    canonical_attempt = output_dir / "attempts" / "render_valid" / "attempt-0001"
    archived_attempt = tmp_path / "archived-scaffold-attempt"
    external_attempt = tmp_path / "external-scaffold-attempt"
    swapped = False

    def swapping_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        nonlocal swapped
        del policy, usd_paths
        canonical_attempt.rename(archived_attempt)
        external_attempt.mkdir()
        canonical_attempt.symlink_to(external_attempt, target_is_directory=True)
        swapped = True
        render_dir = Path(working_dir) / "renders"
        render_dir.mkdir()
        render_path = _write_image(
            render_dir / "toolbox.png",
            (220, 180, 0),
        )
        return {
            "status": "completed",
            "backend": "fake",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "fake",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "images": [str(render_path)],
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        swapping_runtime_renderer,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
    )

    assert swapped is True
    assert run.status.value == "cancelled"
    assert run.result.verdict != "pass"
    assert not any(external_attempt.iterdir())
    assert (archived_attempt / "renders" / "toolbox.png").is_file()


def test_raw_pinned_working_directory_paths_are_canonicalized_before_close(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=RawWorkingDirectoryPathExecutor(source=source),
    )

    assert run.result.verdict == "pass"
    for record in run.checkpoint.records:
        accepted = record.accepted_result
        assert accepted is not None
        expected_attempt_dir = (
            output_dir
            / "attempts"
            / record.template_name
            / f"attempt-{record.attempts:04d}"
        )
        assert {artifact.path for artifact in accepted.evidence_artifacts} == {
            str(expected_attempt_dir / f"{record.template_name}.png")
        }
        assert accepted.result.evidence["image_paths"] == [
            str(expected_attempt_dir / f"{record.template_name}.png")
        ]
        assert accepted.result.evidence_items[0].path == str(
            expected_attempt_dir / f"{record.template_name}.png"
        )
        assert accepted.result.metadata["working_dir"] == str(expected_attempt_dir)


def test_pinned_working_directory_file_uris_are_canonicalized_before_close(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=RawWorkingDirectoryFileUriExecutor(source=source),
    )

    assert run.result.verdict == "pass"
    for record in run.checkpoint.records:
        accepted = record.accepted_result
        assert accepted is not None
        expected_path = (
            output_dir
            / "attempts"
            / record.template_name
            / f"attempt-{record.attempts:04d}"
            / f"{record.template_name}.png"
        )
        assert accepted.result.evidence_items[0].path == str(expected_path)
        assert str(expected_path) in {
            artifact.path for artifact in accepted.evidence_artifacts
        }


def test_checkpoint_cannot_reuse_result_from_stale_attempt(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(first.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["records"][1]["attempts"] = 2
    checkpoint_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"
    assert resumed.checkpoint.records[1].attempts == 3


def test_checkpoint_cannot_drop_workflow_owned_evidence_and_reuse_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(first.checkpoint_path)
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    accepted_payload = checkpoint_payload["records"][1]["accepted_result"]
    result_path = Path(accepted_payload["result_path"])
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload.setdefault("metadata", {}).setdefault("artifacts", {})[
        "report_path"
    ] = str(output_dir / "validation_result.json")
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    accepted_payload["result"] = result_payload
    accepted_payload["result_sha256"] = file_sha256(result_path)
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_missing_declared_evidence_cannot_publish_or_resume_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)

    first = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=MissingTypedEvidenceExecutor(source=source),
    )

    assert first.result.verdict == "fail"
    assert "validation.accepted_evidence_missing" in {
        issue.code for issue in first.result.issues
    }
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert accepted.result.status == "failed"

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "fail"


def test_missing_failed_evidence_cannot_be_downgraded_as_expected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["render.runtime_render_failed"],
            },
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=MissingFailedRenderEvidenceExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    assert "validation.accepted_evidence_missing" in {
        issue.code for issue in run.result.issues
    }
    assert run.result.metadata["workflow_integrity_expected_result_bypassed"] is True


def test_malformed_evidence_path_fails_closed_without_stuck_claim(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)

    first = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=MalformedEvidencePathExecutor(source=source),
    )

    assert first.result.verdict == "fail"
    assert "validation.accepted_evidence_missing" in {
        issue.code for issue in first.result.issues
    }
    assert all(
        record.state != ValidationWorkItemState.RUNNING
        for record in first.checkpoint.records
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "fail"


def test_blank_evidence_path_is_not_accepted_as_attempt_directory(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=BlankEvidencePathExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [""]


def test_blank_typed_evidence_path_is_not_accepted_as_attempt_directory(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=BlankTypedEvidencePathExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [""]


@pytest.mark.parametrize("uri", ("file:", "file://"))
@pytest.mark.parametrize("typed", (False, True))
def test_empty_file_uri_is_not_accepted_as_attempt_directory(
    tmp_path: Path,
    uri: str,
    typed: bool,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=typed,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize(
    "uri",
    (
        "https://example.invalid/evidence.png",
        "s3://bucket/key",
        "file://[::1/bad",
        "file://remote-host",
    ),
)
@pytest.mark.parametrize("typed", (False, True))
def test_remote_evidence_uri_fails_closed(
    tmp_path: Path,
    uri: str,
    typed: bool,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=typed,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize(
    "uri",
    (
        "https://example.invalid/evidence.png",
        "s3://bucket/key",
    ),
)
def test_typed_artifact_map_remote_uri_fails_closed(
    tmp_path: Path,
    uri: str,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=False,
            named_map_typed_metadata=True,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize("artifact_name", ("never-written.dds", "never-written"))
@pytest.mark.parametrize("container_shape", ("map", "record", "nested_record"))
def test_typed_artifact_container_missing_value_fails_closed(
    tmp_path: Path,
    artifact_name: str,
    container_shape: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=artifact_name,
            typed=False,
            named_map_typed_metadata=container_shape == "map",
            named_record_typed_metadata=container_shape == "record",
            named_nested_record_typed_metadata=container_shape == "nested_record",
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                output_dir / "attempts" / "look_right" / "attempt-0001" / artifact_name
            ).resolve()
        ),
    ]


def test_typed_artifact_map_missing_escape_fails_closed_with_spaced_base(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run with spaces"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri="../ghost.png",
            typed=False,
            named_map_typed_metadata=True,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str((output_dir / "attempts" / "look_right" / "ghost.png").resolve()),
    ]


@pytest.mark.parametrize(
    "uri",
    (
        "file://[::1/bad",
        "file://remote-host",
        "s3://bucket/proof.png",
    ),
)
def test_typed_scene_metadata_invalid_file_authority_fails_closed(
    tmp_path: Path,
    uri: str,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=False,
            scene_typed_metadata=True,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize("scene_semantic", (False, True))
def test_typed_metadata_nonlocal_file_authority_fails_closed(
    tmp_path: Path,
    scene_semantic: bool,
) -> None:
    source, reference = _inputs(tmp_path)
    local_evidence = _write_image(
        tmp_path / "local-proof.png",
        (20, 40, 60),
    )
    uri = f"file://remote-host{local_evidence.resolve()}"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=False,
            typed_metadata=not scene_semantic,
            scene_typed_metadata=scene_semantic,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize("scene_semantic", (False, True))
def test_typed_metadata_localhost_file_authority_is_case_insensitive(
    tmp_path: Path,
    scene_semantic: bool,
) -> None:
    source, reference = _inputs(tmp_path)
    local_evidence = _write_image(
        tmp_path / "local-proof.png",
        (20, 40, 60),
    )
    uri = f"file://LOCALHOST{local_evidence.resolve()}"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=uri,
            typed=False,
            typed_metadata=not scene_semantic,
            scene_typed_metadata=scene_semantic,
        ),
    )

    assert run.result.verdict == "pass"
    accepted = run.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert str(local_evidence.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }


def test_typed_scene_metadata_missing_local_file_uri_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    missing_evidence = tmp_path / "missing scene proof.custom"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=missing_evidence.as_uri(),
            typed=False,
            scene_typed_metadata=True,
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(missing_evidence.resolve())
    ]


def test_ordinary_metadata_names_are_not_treated_as_artifact_paths(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=OrdinaryMetadataExecutor(source=source),
    )

    assert run.result.verdict == "pass"
    look_right = run.checkpoint.records[1].accepted_result
    assert look_right is not None
    assert look_right.result.metadata == {
        "executor": "fake",
        "security": "passed",
        "maturity": "stable",
        "pathology": "none",
    }


def test_top_level_role_suffixed_prose_is_not_treated_as_artifact_paths(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    prose = {
        "final_render": "confidence 0.87",
        "status_report": "ok",
    }

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata=prose,
        ),
    )

    assert run.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in run.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert accepted.result.metadata == {
        "executor": "fake",
        **prose,
    }
    evidence_paths = {artifact.path for artifact in accepted.evidence_artifacts}
    assert set(prose.values()).isdisjoint(evidence_paths)


@pytest.mark.parametrize(
    "path_kind",
    ("remote_uri", "relative_custom", "file_uri"),
)
def test_top_level_role_suffixed_explicit_path_syntax_fails_closed(
    tmp_path: Path,
    path_kind: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    if path_kind == "remote_uri":
        declared_path = "s3://bucket/proof.png"
        expected_path = declared_path
    elif path_kind == "relative_custom":
        declared_path = "renders/proof.custom"
        expected_path = str(
            (
                output_dir / "attempts" / "look_right" / "attempt-0001" / declared_path
            ).resolve()
        )
    else:
        missing_evidence = tmp_path / "missing proof.custom"
        declared_path = missing_evidence.as_uri()
        expected_path = str(missing_evidence.resolve())

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"final_render": declared_path},
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [expected_path]


def test_top_level_role_suffixed_existing_path_is_sealed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    external_evidence = _write_image(
        tmp_path / "final render.custom",
        (20, 40, 60),
        image_format="PNG",
    )
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"final_render": str(external_evidence)},
        ),
    )

    assert first.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in first.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert str(external_evidence) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    external_evidence.unlink()
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_top_level_authoritative_sequence_record_is_sealed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    external_evidence = _write_image(
        tmp_path / "sequence proof.custom",
        (20, 40, 60),
        image_format="PNG",
    )
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": [
                    {
                        "path": str(external_evidence),
                        "code": "visual.accepted",
                        "severity": "info",
                        "warnings": ["camera fallback used"],
                        "issues": [
                            {
                                "code": "visual.accepted",
                                "severity": "info",
                                "message": "Evidence accepted",
                            },
                        ],
                        "details": {
                            "render_response": {
                                "status": "ok",
                                "preview": str(external_evidence),
                            },
                        },
                    },
                ],
            },
        ),
    )

    assert first.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in first.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert str(external_evidence) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    external_evidence.unlink()
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_nested_render_output_dir_is_context_not_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    context_evidence = _write_image(
        tmp_path / "context-proof.png",
        (20, 40, 60),
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "runtime_context": {"render_output_dir": "."},
                "artifacts": {
                    "context": {
                        "final_render": "passed",
                        "status_report": "ok",
                        "camera_path": "/World/Camera",
                        "preview": str(context_evidence),
                    },
                    "runtime_context": {
                        "final_render": "passed",
                        "status_report": "ok",
                        "artifacts": [
                            {
                                "path": str(context_evidence),
                                "code": "visual.accepted",
                                "severity": "info",
                            },
                        ],
                    },
                    "metadata": {
                        "runtime": {
                            "render_output_dir": ".",
                        },
                        "final_render": "passed",
                        "status_report": "ok",
                    },
                },
            },
        ),
    )

    assert run.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in run.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert accepted.result.metadata["runtime_context"]["render_output_dir"] == "."
    assert accepted.result.metadata["artifacts"]["context"]["final_render"] == "passed"
    assert accepted.result.metadata["artifacts"]["context"]["status_report"] == "ok"
    assert (
        accepted.result.metadata["artifacts"]["context"]["camera_path"]
        == "/World/Camera"
    )
    assert (
        accepted.result.metadata["artifacts"]["runtime_context"]["final_render"]
        == "passed"
    )
    assert (
        accepted.result.metadata["artifacts"]["runtime_context"]["status_report"]
        == "ok"
    )
    assert (
        accepted.result.metadata["artifacts"]["metadata"]["runtime"][
            "render_output_dir"
        ]
        == "."
    )
    assert accepted.result.metadata["artifacts"]["metadata"]["final_render"] == "passed"
    assert accepted.result.metadata["artifacts"]["metadata"]["status_report"] == "ok"
    attempt_dir = Path(accepted.result_path).parent.resolve()
    assert str(attempt_dir) not in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    assert str(context_evidence) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }


def test_context_scene_semantic_missing_file_uri_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    missing_evidence = tmp_path / "missing context proof.custom"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "context": {
                        "camera_path": missing_evidence.as_uri(),
                    },
                },
            },
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(missing_evidence.resolve())
    ]


def test_scene_semantic_tilde_path_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    declared_path = "~/validation-agent-wave2-missing-proof.custom"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "camera_path": declared_path,
                },
            },
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(Path(declared_path).expanduser().resolve())
    ]


def test_unknown_user_tilde_evidence_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    declared_path = "~validation-agent-wave2-user-does-not-exist/proof.custom"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "report_paths": [declared_path],
                },
            },
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [declared_path]
    assert all(record.state != "running" for record in run.checkpoint.records)


@pytest.mark.parametrize("structured", (False, True))
def test_typed_artifact_source_remote_uri_fails_closed(
    tmp_path: Path,
    structured: bool,
) -> None:
    source, reference = _inputs(tmp_path)
    uri = "s3://bucket/proof.png"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "source": {"uri": uri} if structured else uri,
                }
            },
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [uri]


@pytest.mark.parametrize(
    "source_path",
    (
        "missing proof.png",
        "proofs/missing evidence.custom",
        "proofs/missing evidence",
    ),
)
def test_typed_artifact_source_missing_path_syntax_fails_closed(
    tmp_path: Path,
    source_path: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"source": source_path}},
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                output_dir / "attempts" / "look_right" / "attempt-0001" / source_path
            ).resolve()
        ),
    ]


@pytest.mark.parametrize("source", ("renderer-map.v2", "qwen2.5-vl", "gpt-4.1"))
def test_typed_artifact_source_dotted_identifier_is_metadata(
    tmp_path: Path,
    source: str,
) -> None:
    source_asset, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source_asset, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source_asset,
            metadata={
                "artifacts": {
                    "metadata": {
                        "source": source,
                    },
                },
            },
        ),
    )

    assert run.result.verdict == "pass"
    assert "validation.accepted_evidence_missing" not in {
        issue.code for issue in run.result.issues
    }


def test_typed_artifact_source_bare_custom_filename_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"source": "proof.custom"}},
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                output_dir / "attempts" / "look_right" / "attempt-0001" / "proof.custom"
            ).resolve()
        )
    ]


def test_typed_artifact_source_sequence_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"source": ["proof.custom"]}},
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                output_dir / "attempts" / "look_right" / "attempt-0001" / "proof.custom"
            ).resolve()
        )
    ]


def test_typed_artifact_nested_metadata_source_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": [
                    {
                        "metadata": {
                            "source": "/absolute/missing-proof.png",
                        }
                    }
                ]
            },
        ),
    )

    assert run.result.verdict == "fail"
    assert "validation.accepted_evidence_missing" in {
        issue.code for issue in run.result.issues
    }


def test_typed_artifact_nested_context_derivative_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": [
                    {
                        "metadata": {
                            "derivatives": [
                                {
                                    "front": "proof.png",
                                    "caption": "Front view proof",
                                }
                            ]
                        }
                    }
                ]
            },
        ),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                tmp_path
                / "run"
                / "attempts"
                / "look_right"
                / "attempt-0001"
                / "proof.png"
            ).resolve()
        )
    ]


def test_typed_artifact_source_dotted_prose_is_metadata(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    prose = "renderer produced qwen2.5-vl"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"source": prose}},
        ),
    )

    assert run.result.verdict == "pass"
    accepted = run.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert accepted.result.metadata["artifacts"]["source"] == prose


def test_authoritative_render_output_dir_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_dir = tmp_path / "external-render-output"
    evidence_dir.mkdir()
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "artifact_paths": {
                        "metadata": {
                            "render_output_dir": str(evidence_dir),
                        },
                    },
                },
            },
        ),
    )

    assert first.result.verdict == "pass"
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert str(evidence_dir.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    evidence_dir.rmdir()
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_authoritative_artifact_path_in_sequence_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_dir = tmp_path / "external-render-output"
    evidence_dir.mkdir()
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={
                "artifacts": {
                    "artifact_paths": {
                        "metadata": [
                            {
                                "render_output_dir": str(evidence_dir),
                            },
                        ],
                    },
                },
            },
        ),
    )

    assert first.result.verdict == "pass"
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert str(evidence_dir.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    evidence_dir.rmdir()
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


@pytest.mark.parametrize("metadata_kind", ("artifact_source", "scene_semantic"))
def test_typed_windows_drive_path_is_treated_as_local_evidence(
    tmp_path: Path,
    metadata_kind: str,
) -> None:
    source, reference = _inputs(tmp_path)
    windows_path = r"C:\validation evidence\missing proof.custom"
    output_dir = tmp_path / "run"
    if metadata_kind == "artifact_source":
        executor = StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"source": windows_path}},
        )
    else:
        executor = UriEvidenceExecutor(
            source=source,
            uri=windows_path,
            typed=False,
            scene_typed_metadata=True,
        )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    expected_path = Path(windows_path)
    if not expected_path.is_absolute():
        expected_path = (
            output_dir / "attempts" / "look_right" / "attempt-0001" / expected_path
        )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(expected_path.resolve())
    ]


def test_suffixed_prose_roles_are_not_treated_as_artifact_paths(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    prose = {
        "render_command": "render output.png",
        "error_message": "could not write output.png",
        "failure_reason": "renderer rejected output.png",
        "review_note": "see output.png",
        "review_notes": "see output.png",
        "error_title": "missing output.png",
        "image_alt_text": "rendered output.png",
        "display_text": "rendered output.png",
        "artifact_id": "run.1",
        "id": "run.1",
        "sha": "sha256:not-an-artifact",
        "content_sha": "sha256:not-an-artifact",
        "checksum": "sha256:not-an-artifact",
        "content_checksum": "sha256:not-an-artifact",
        "content_sha256": "sha256:not-an-artifact",
        "code": "visual.accepted",
        "severity": "info",
        "payload_digest": "sha256:not-an-artifact",
        "render_source": "renderer",
        "front_camera": "/World/Camera",
        "render_cameras": ["/World/Camera", "/World/RearCamera"],
        "shas": ["sha256:not-an-artifact"],
        "summaries": ["front view accepted"],
    }

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": prose},
        ),
    )

    assert run.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in run.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert accepted.result.metadata["artifacts"] == prose
    evidence_paths = {artifact.path for artifact in accepted.evidence_artifacts}
    prose_values = {
        item
        for value in prose.values()
        for item in (value if isinstance(value, list) else [value])
    }
    assert prose_values.isdisjoint(evidence_paths)


@pytest.mark.parametrize(
    "artifact_name",
    (
        "front view.png",
        "front view.custom",
        "front view",
    ),
)
def test_existing_external_scene_evidence_with_spaces_is_sealed(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    external_evidence = _write_image(
        tmp_path / artifact_name,
        (20, 40, 60),
        image_format="PNG",
    )
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=StructuredMetadataEvidenceExecutor(
            source=source,
            metadata={"artifacts": {"camera_path": str(external_evidence)}},
        ),
    )

    assert first.result.verdict == "pass"
    accepted = next(
        record.accepted_result
        for record in first.checkpoint.records
        if record.template_name == "look_right"
    )
    assert accepted is not None
    assert str(external_evidence) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    external_evidence.unlink()
    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_missing_named_map_artifact_with_unlisted_suffix_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=MissingMapEvidenceExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                tmp_path
                / "run"
                / "attempts"
                / "look_right"
                / "attempt-0001"
                / "never-written.webp"
            ).resolve()
        )
    ]


def test_workflow_owned_artifact_cannot_be_accepted_as_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=WorkflowArtifactEvidenceExecutor(source=source),
    )

    assert first.result.verdict == "fail"
    assert "validation.workflow_artifact_evidence_collision" in {
        issue.code for issue in first.result.issues
    }
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert accepted.result.status == "failed"
    accepted_evidence_paths = {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    assert str((output_dir / "validation_result.json").resolve()) not in (
        accepted_evidence_paths
    )
    assert (
        str(
            (
                output_dir
                / "attempts"
                / "render_valid"
                / "attempt-0001"
                / "template_result.json"
            ).resolve()
        )
        not in accepted_evidence_paths
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "fail"


def test_workflow_owned_artifact_hard_link_cannot_be_accepted_as_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_path = tmp_path / "external-hard-link-evidence.json"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=WorkflowArtifactHardLinkEvidenceExecutor(
            source=source,
            evidence_path=evidence_path,
        ),
    )

    assert run.result.verdict == "fail"
    assert "validation.workflow_artifact_evidence_collision" in {
        issue.code for issue in run.result.issues
    }
    accepted = run.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert str(evidence_path.resolve()) not in {
        artifact.path for artifact in accepted.evidence_artifacts
    }


def test_executor_cannot_mutate_sealed_request_or_plan(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=MutatingContextExecutor(source=source),
    )

    assert first.result.verdict == "pass"
    assert "executor_mutation" not in request.policy
    assert "executor_mutation" not in first.request.policy
    assert "executor_mutation" not in first.plan.metadata
    assert all("executor_mutation" not in step.metadata for step in first.plan.steps)
    persisted_request = json.loads(Path(first.request_path).read_text(encoding="utf-8"))
    persisted_plan = json.loads(Path(first.plan_path).read_text(encoding="utf-8"))
    assert "executor_mutation" not in persisted_request["policy"]
    assert "executor_mutation" not in persisted_plan["metadata"]
    assert all(
        "executor_mutation" not in step["metadata"] for step in persisted_plan["steps"]
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "pass"


def test_evidence_directory_containing_workflow_output_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_dir = tmp_path / "evidence"
    output_dir = evidence_dir / "run"
    checkpoint_path = tmp_path / "checkpoint-state" / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    request = _request(source, reference)

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=WorkflowAncestorEvidenceExecutor(
            source=source,
            evidence_dir=evidence_dir,
        ),
        checkpoint_store=store,
    )

    assert first.result.verdict == "fail"
    assert "validation.workflow_artifact_evidence_collision" in {
        issue.code for issue in first.result.issues
    }
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert str(evidence_dir.resolve()) not in {
        artifact.path for artifact in accepted.evidence_artifacts
    }

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        checkpoint_store=store,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "fail"


def test_current_attempt_directory_cannot_be_accepted_as_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UriEvidenceExecutor(
            source=source,
            uri=".",
            typed=False,
            named_map_typed_metadata=True,
        ),
    )

    assert run.result.verdict == "fail"
    assert "validation.workflow_artifact_evidence_collision" in {
        issue.code for issue in run.result.issues
    }
    accepted = run.checkpoint.records[1].accepted_result
    assert accepted is not None
    assert all(artifact.kind != "directory" for artifact in accepted.evidence_artifacts)


def test_typed_evidence_metadata_artifacts_are_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    assert first.result.verdict == "pass"
    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    supplemental_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-metadata.png"
    )
    thumbnail_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-thumbnail.png"
    )
    structured_image_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-structured-image.png"
    )
    nested_detail_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-detail.png"
    )
    structured_sequence_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-structured-sequence.png"
    )
    image_map_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-image-map.png"
    )
    nested_sequence_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-sequence.png"
    )
    mixed_image_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-mixed-image.png"
    )
    bare_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-bare-path.png"
    )
    bare_paths_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-bare-paths.png"
    )
    video_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-video-path.png"
    )
    depth_map_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-depth-map-path.png"
    )
    texture_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-texture-path.png"
    )
    named_evidence_map_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-named-evidence-map.png"
    )
    named_image_map_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-named-image-map.png"
    )
    custom_image_map_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-custom-image-map.png"
    )
    extensionless_image_map_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-custom-image-extensionless"
    )
    extensionless_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-extensionless"
    )
    generic_extensionless_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-generic-extensionless"
    )
    camera_label_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-camera-label.png"
    )
    mesh_preview_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-mesh-preview.png"
    )
    checksum_role_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-checksum-role.png"
    )
    nested_record_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-record.png"
    )
    nested_capture_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-capture.png"
    )
    plural_record_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-plural-record.png"
    )
    structured_artifact_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-structured-artifact.png"
    )
    structured_thumbnail_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-structured-thumbnail.png"
    )
    summary_report_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-summary-report.png"
    )
    record_diff_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-record-diff.png"
    )
    top_plural_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-top-plural.png"
    )
    deep_report_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-deep-report.png"
    )
    producer_record_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-producer-record.png"
    )
    plural_manifest_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-plural-manifest.png"
    )
    sequence_metadata_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-sequence-metadata.png"
    )
    contact_sheet_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-contact-sheet.png"
    )
    overlay_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-overlay.png"
    )
    mask_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-mask.png"
    )
    heatmap_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-heatmap.png"
    )
    grounding_packet_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-grounding-packet.png"
    )
    hybrid_front_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-hybrid-front.png"
    )
    hybrid_side_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-hybrid-side.png"
    )
    node_preview_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-node-preview.png"
    )
    map_camera_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-map-camera.png"
    )
    record_camera_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-record-camera.png"
    )
    scene_extensionless_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-scene-extensionless"
    )
    sequence_camera_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-sequence-camera.png"
    )
    root_scene_file_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-root-scene-file.usda"
    )
    node_map_preview_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-node-map-preview.png"
    )
    scene_record_uri_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-scene-record-uri.png"
    )
    record_front_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-record-front.png"
    )
    root_artifact_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-root-artifact.png"
    )
    scene_record_camera_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-scene-record-camera.png"
    )
    evidence_bundle_dir = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-evidence-bundle"
    )
    nested_derivative_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-derivative.png"
    )
    nested_metadata_artifact_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-metadata-artifact.png"
    )
    source_role_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-source-role.png"
    )
    evidence_paths = {artifact.path for artifact in accepted.evidence_artifacts}
    assert {
        str(supplemental_path.resolve()),
        str(thumbnail_path.resolve()),
        str(structured_image_path.resolve()),
        str(nested_detail_path.resolve()),
        str(structured_sequence_path.resolve()),
        str(image_map_path.resolve()),
        str(nested_sequence_path.resolve()),
        str(mixed_image_path.resolve()),
        str(bare_path.resolve()),
        str(bare_paths_path.resolve()),
        str(video_path.resolve()),
        str(depth_map_path.resolve()),
        str(texture_path.resolve()),
        str(named_evidence_map_path.resolve()),
        str(named_image_map_path.resolve()),
        str(custom_image_map_path.resolve()),
        str(extensionless_image_map_path.resolve()),
        str(extensionless_path.resolve()),
        str(generic_extensionless_path.resolve()),
        str(camera_label_path.resolve()),
        str(mesh_preview_path.resolve()),
        str(checksum_role_path.resolve()),
        str(nested_record_path.resolve()),
        str(nested_capture_path.resolve()),
        str(plural_record_path.resolve()),
        str(structured_artifact_path.resolve()),
        str(structured_thumbnail_path.resolve()),
        str(summary_report_path.resolve()),
        str(record_diff_path.resolve()),
        str(top_plural_path.resolve()),
        str(deep_report_path.resolve()),
        str(producer_record_path.resolve()),
        str(plural_manifest_path.resolve()),
        str(sequence_metadata_path.resolve()),
        str(contact_sheet_path.resolve()),
        str(overlay_path.resolve()),
        str(mask_path.resolve()),
        str(heatmap_path.resolve()),
        str(grounding_packet_path.resolve()),
        str(hybrid_front_path.resolve()),
        str(hybrid_side_path.resolve()),
        str(node_preview_path.resolve()),
        str(map_camera_path.resolve()),
        str(record_camera_path.resolve()),
        str(scene_extensionless_path.resolve()),
        str(sequence_camera_path.resolve()),
        str(root_scene_file_path.resolve()),
        str(node_map_preview_path.resolve()),
        str(scene_record_uri_path.resolve()),
        str(record_front_path.resolve()),
        str(root_artifact_path.resolve()),
        str(scene_record_camera_path.resolve()),
        str(evidence_bundle_dir.resolve()),
        str(nested_derivative_path.resolve()),
        str(nested_metadata_artifact_path.resolve()),
        str(source_role_path.resolve()),
    } <= evidence_paths
    assert {
        "Front view",
        "Reference comparison thumbnail",
        "Side view",
        "Close-up",
        "Front map view",
        "image/png",
        "image",
        "+x",
        "+y",
        "+z",
        "renderer",
        "renderer-map",
        "renderer-artifact-map",
        "Artifact bundle overview",
        "Plural artifact collection",
        "Front image description",
        "Front views",
        "passed",
        "24 fps",
        "sha256:not-an-artifact-path",
        "Structured artifact record",
        "Plural artifact record",
        "Deep report overview",
        "Named evidence thumbnail",
        "QA manifest overview",
        "QA report overview",
        "renderer",
        "renderer-direct",
        "hybrid-renderer",
        "Hybrid views",
        "run.1",
        "typed images id.png",
        "sha256:inline-metadata",
        "Rendered front, side. QA pass",
        "sRGB",
        "right-handed",
        "front",
        "side",
        "front.v2",
        "side.v2",
        "qwen2.5-vl",
        "nvidia.nim",
        "rendered to output.png",
        "render completed",
        "reference comparison passed",
        "render finished",
        "../missing-shot.usd",
        "2s",
        "0.3s",
        "validator",
    }.isdisjoint(evidence_paths)
    assert (
        not {
            "/World/Looks/Paint",
            "/World/Looks/ArtifactMetadata",
            "/World/Shader.png",
            "/World/Cube.size",
            "/World/Cam_v1.2",
            "/World/Looks",
            "/World",
            "/World/Body",
            "/World/SourceBody",
            "/World/Camera",
            "/World/Joints/Hinge",
            "/World/Skeleton",
            "/Root/Body/Collision",
            "/asset/body",
            "/root/body",
            "/asset/body/collider",
            "/World/Body.visibility",
            "/World/Body/Mesh",
            "/World/Body/Child",
            "/World/Camera/NamedEvidence",
            "/World/NamedEvidenceSubject",
            "/World/NestedRecordCamera",
            "/World/PhysicsScene",
            "/World/Collection",
            "/World/RenderMesh",
            "/World/Mesh.primvars:st",
            "/World/Cameras/TextureAgentFinal",
            "/World/Looks/Paint/Shader",
            "/World/Looks/Paint/Shader/Node",
            "/World/ResultShader",
            "/World/ResultShader/Node",
            "/World/Looks/StructuredShader",
            "/World/N",
            "/World/RecordEvidenceCamera",
            "/World/RecordImageCamera",
            "/World/Target",
            "/World/Mesh/face_0",
            "/",
        }
        & evidence_paths
    )
    nested_detail_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


@pytest.mark.parametrize(
    "artifact_name",
    (
        "typed-bare-path.png",
        "typed-bare-paths.png",
        "typed-video-path.png",
        "typed-depth-map-path.png",
        "typed-texture-path.png",
        "typed-named-evidence-map.png",
        "typed-named-image-map.png",
        "typed-custom-image-map.png",
        "typed-custom-image-extensionless",
        "typed-extensionless",
        "typed-generic-extensionless",
        "typed-camera-label.png",
        "typed-mesh-preview.png",
        "typed-checksum-role.png",
        "typed-nested-record.png",
        "typed-nested-capture.png",
        "typed-plural-record.png",
        "typed-structured-artifact.png",
        "typed-structured-thumbnail.png",
        "typed-summary-report.png",
        "typed-record-diff.png",
        "typed-top-plural.png",
        "typed-deep-report.png",
        "typed-producer-record.png",
        "typed-plural-manifest.png",
        "typed-sequence-metadata.png",
        "typed-contact-sheet.png",
        "typed-overlay.png",
        "typed-mask.png",
        "typed-heatmap.png",
        "typed-grounding-packet.png",
        "typed-hybrid-front.png",
        "typed-hybrid-side.png",
        "typed-node-preview.png",
        "typed-map-camera.png",
        "typed-record-camera.png",
        "typed-scene-extensionless",
        "typed-sequence-camera.png",
        "typed-root-scene-file.usda",
        "typed-node-map-preview.png",
        "typed-scene-record-uri.png",
        "typed-record-front.png",
        "typed-root-artifact.png",
        "typed-scene-record-camera.png",
        "typed-nested-derivative.png",
        "typed-nested-metadata-artifact.png",
        "typed-source-role.png",
    ),
)
def test_typed_evidence_metadata_path_variants_are_sealed_for_resume(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    artifact_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / artifact_name
    )
    assert str(artifact_path.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    artifact_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_typed_evidence_metadata_nested_sequence_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    nested_sequence_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-nested-sequence.png"
    )
    assert str(nested_sequence_path.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    nested_sequence_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_typed_evidence_record_mapping_metadata_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=NestedRecordMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    nested_path = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "nested-record-metadata.png"
    )
    assert str(nested_path.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    nested_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_typed_evidence_metadata_directory_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    bundle_dir = (
        output_dir
        / "attempts"
        / "look_right"
        / "attempt-0001"
        / "typed-evidence-bundle"
    )
    assert str(bundle_dir.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    bundle_dir.rmdir()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_typed_evidence_directory_with_symlink_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=SymlinkDirectoryEvidenceExecutor(source=source),
    )

    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_missing"
    )
    assert missing_issue.details["missing_evidence_paths"] == [
        str(
            (
                output_dir
                / "attempts"
                / "look_right"
                / "attempt-0001"
                / "symlink-evidence"
            ).resolve()
        ),
    ]


@pytest.mark.parametrize("directory_role", ("source", "reference"))
def test_identity_directory_with_symlink_is_rejected(
    tmp_path: Path,
    directory_role: str,
) -> None:
    source = _write_source(tmp_path / "toolbox.usda")
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    target_dir = tmp_path / "linked-target"
    target_dir.mkdir()
    (target_dir / "content.txt").write_text("mutable\n", encoding="utf-8")
    identity_dir = tmp_path / f"{directory_role}-directory"
    identity_dir.mkdir()
    (identity_dir / "linked-content").symlink_to(
        target_dir,
        target_is_directory=True,
    )
    if directory_role == "source":
        _write_source(identity_dir / "asset.usda")
        request = _request(identity_dir, reference)
    else:
        _write_image(identity_dir / "reference.png", (220, 180, 0))
        request = _request(source, identity_dir)

    with pytest.raises(
        ValueError,
        match="Artifact directory cannot contain symlinks",
    ):
        run_validation_workflow(
            request,
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )


def test_typed_evidence_metadata_artifact_map_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    thumbnail_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-thumbnail.png"
    )
    assert str(thumbnail_path.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    thumbnail_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_typed_evidence_metadata_image_map_is_sealed_for_resume(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    accepted = first.checkpoint.records[1].accepted_result
    assert accepted is not None
    image_map_path = (
        output_dir / "attempts" / "look_right" / "attempt-0001" / "typed-image-map.png"
    )
    assert str(image_map_path.resolve()) in {
        artifact.path for artifact in accepted.evidence_artifacts
    }
    image_map_path.unlink()

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_resume_accepts_prior_checkpoint_evidence_order(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    assert first.result.verdict == "pass"
    checkpoint_path = output_dir / "validation_checkpoint.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    typed_evidence = checkpoint_payload["records"][1]["accepted_result"][
        "evidence_artifacts"
    ]
    assert len(typed_evidence) > 1
    checkpoint_payload["records"][1]["accepted_result"]["evidence_artifacts"] = list(
        reversed(typed_evidence)
    )
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == []
    assert resumed.result.verdict == "pass"


def test_resume_invalidates_legacy_checkpoint_with_missing_metadata_evidence(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"

    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=TypedMetadataEvidenceExecutor(source=source),
    )

    assert first.result.verdict == "pass"
    checkpoint_path = output_dir / "validation_checkpoint.json"
    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    accepted_payload = checkpoint_payload["records"][1]["accepted_result"]
    primary_path = accepted_payload["result"]["evidence_items"][0]["path"]
    primary_artifacts = [
        artifact
        for artifact in accepted_payload["evidence_artifacts"]
        if artifact["path"] == primary_path
    ]
    metadata_artifacts = [
        artifact
        for artifact in accepted_payload["evidence_artifacts"]
        if artifact["path"] != primary_path
    ]
    assert len(primary_artifacts) == 1
    assert metadata_artifacts
    for artifact in metadata_artifacts:
        artifact_path = Path(artifact["path"])
        if artifact["kind"] == "directory":
            artifact_path.rmdir()
        else:
            artifact_path.unlink()
    accepted_payload["evidence_artifacts"] = primary_artifacts
    checkpoint_path.write_text(
        json.dumps(checkpoint_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resume_executor = FakeValidationExecutor(source=source)
    resumed = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=resume_executor,
        resume=True,
    )

    assert resume_executor.calls == ["look_right"]
    assert resumed.result.verdict == "pass"


def test_missing_reference_blocks_look_right_without_invoking_it(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "toolbox.usda")
    missing_reference = tmp_path / "missing-reference.png"
    executor = FakeValidationExecutor(source=source)

    run = run_validation_workflow(
        _request(
            source,
            missing_reference,
            policy={"gate_policy": {"dependency_unavailable": "block"}},
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "fail"
    assert "visual.reference_evidence_missing" in {
        issue.code for issue in run.result.issues
    }


def test_required_but_unconfigured_reference_blocks_look_right(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference).model_copy(
        update={
            "policy": {
                "visual_evidence_mode": "canonical_usd",
                "reference_evidence_required": True,
                "expected_verdict": "fail",
                "expected_issue_codes": ["visual.reference_evidence_missing"],
            }
        }
    )
    executor = FakeValidationExecutor(source=source)

    run = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "fail"
    missing_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "visual.reference_evidence_missing"
    )
    assert missing_issue.severity == "fail"
    assert run.result.metadata["workflow_integrity_expected_result_bypassed"] is True


def test_reference_changed_after_render_blocks_judge(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(source=source)
    reference_changed = False

    def change_reference_after_render(checkpoint: object) -> None:
        nonlocal reference_changed
        records = getattr(checkpoint, "records")
        if not reference_changed and records[0].accepted_result is not None:
            reference_changed = True
            _write_image(reference, (4, 5, 6))

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
        progress_callback=change_reference_after_render,
    )

    assert reference_changed is True
    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "fail"
    assert "validation.reference_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


def test_reference_changed_during_judge_rejects_late_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = MutatingReferenceExecutor(
        source=source,
        reference=reference,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid", "look_right"]
    assert run.result.verdict == "fail"
    assert "validation.reference_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


def test_render_evidence_changed_during_judge_rejects_late_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = MutatingRenderEvidenceExecutor(source=source)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid", "look_right"]
    assert run.result.verdict == "fail"
    assert "validation.render_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


def test_passed_render_with_missing_image_fails_closed(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = MissingRenderEvidenceExecutor(source=source)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "fail"
    assert "validation.render_evidence_missing" in {
        issue.code for issue in run.result.issues
    }


@pytest.mark.parametrize("hard_link", (False, True))
def test_render_evidence_cannot_alias_reference_input(
    tmp_path: Path,
    hard_link: bool,
) -> None:
    source, reference = _inputs(tmp_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=InputAliasingRenderEvidenceExecutor(
            source=source,
            input_path=reference,
            hard_link=hard_link,
        ),
    )

    assert run.result.verdict == "fail"
    assert "validation.render_evidence_input_collision" in {
        issue.code for issue in run.result.issues
    }


def test_render_evidence_cannot_hard_link_input_directory_member(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    source_member = _write_image(source_dir / "existing.png", (12, 34, 56))
    reference = _write_image(tmp_path / "reference.png", (80, 80, 80))

    run = run_validation_workflow(
        _request(source_dir, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=InputAliasingRenderEvidenceExecutor(
            source=source_dir,
            input_path=source_member,
            hard_link=True,
        ),
    )

    assert run.result.verdict == "fail"
    assert "validation.render_evidence_input_collision" in {
        issue.code for issue in run.result.issues
    }


def test_render_evidence_aliased_after_initial_validation_cannot_publish_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(source=source)
    original_evidence_artifacts = cast(
        Callable[..., Any],
        getattr(workflow_module, "_evidence_artifacts"),
    )
    aliased = False

    def alias_before_capture(
        result: ValidationTemplateResult,
        *,
        base_dir: Path,
    ) -> Any:
        nonlocal aliased
        if result.template_name == "render_valid" and not aliased:
            render_path = Path(result.evidence["image_paths"][0])
            render_path.unlink()
            render_path.hardlink_to(reference)
            aliased = True
        return original_evidence_artifacts(result, base_dir=base_dir)

    monkeypatch.setattr(
        workflow_module,
        "_evidence_artifacts",
        alias_before_capture,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert aliased is True
    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "fail"
    assert "validation.accepted_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


def test_dependency_unavailable_never_becomes_a_silent_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(
        source=source,
        dependency_unavailable=True,
    )

    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={"gate_policy": {"dependency_unavailable": "block"}},
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    assert "validation.dependency_unavailable" in {
        issue.code for issue in run.result.issues
    }


def test_failed_render_skips_judge_without_claiming_integrity_failure(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = NonpassingRenderExecutor(source=source, status="failed")

    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["render.runtime_render_failed"],
            },
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "warn"
    assert [result.status for result in run.result.template_results] == [
        "warn",
        "skipped",
    ]
    issue_codes = {issue.code for issue in run.result.issues}
    assert "render.runtime_render_failed" in issue_codes
    assert "validation.render_evidence_stale" not in issue_codes
    assert run.result.metadata["expected_result"]["matched"] is True
    assert "workflow_integrity_expected_result_bypassed" not in run.result.metadata


@pytest.mark.parametrize(
    ("status", "expected_verdict"),
    (
        ("error", "fail"),
        ("warn", "warn"),
        ("needs_refinement", "needs_refinement"),
    ),
)
def test_nonpassing_render_skips_judge_without_claiming_stale_evidence(
    tmp_path: Path,
    status: Literal["error", "warn", "needs_refinement"],
    expected_verdict: Literal["fail", "warn", "needs_refinement"],
) -> None:
    source, reference = _inputs(tmp_path)
    executor = NonpassingRenderExecutor(source=source, status=status)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == expected_verdict
    assert [result.status for result in run.result.template_results] == [
        status,
        "skipped",
    ]
    assert "validation.render_evidence_stale" not in {
        issue.code for issue in run.result.issues
    }


def test_dependency_unavailable_preserves_warning_without_block_policy(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(
        source=source,
        dependency_unavailable=True,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "warn"
    assert [result.status for result in run.result.template_results] == [
        "skipped",
        "skipped",
    ]
    issue_codes = {issue.code for issue in run.result.issues}
    assert "render.renderer_unavailable" in issue_codes
    assert "validation.render_evidence_stale" not in issue_codes
    assert "validation.dependency_unavailable" not in issue_codes


def test_render_only_pass_does_not_claim_visual_reference_match(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(source=source)

    run = run_validation_workflow(
        _request(
            source,
            reference,
            requested_templates=("render_valid",),
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert executor.calls == ["render_valid"]
    assert run.result.verdict == "pass"
    recommendation = run.result.recommended_action or ""
    assert "no visual-reference comparison was requested" in recommendation
    assert "matched the supplied visual reference" not in recommendation


def test_render_only_run_ignores_unused_reference_drift(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    reference_changed = False

    def change_unused_reference_after_render(checkpoint: object) -> None:
        nonlocal reference_changed
        records = getattr(checkpoint, "records")
        if reference_changed or records[0].accepted_result is None:
            return
        _write_image(reference, (1, 2, 3))
        reference_changed = True

    run = run_validation_workflow(
        _request(
            source,
            reference,
            requested_templates=("render_valid",),
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        progress_callback=change_unused_reference_after_render,
    )

    assert reference_changed is True
    assert run.result.verdict == "pass"
    assert run.checkpoint.workflow_identity.reference_artifacts == ()
    assert "validation.reference_evidence_stale" not in {
        issue.code for issue in run.result.issues
    }


def test_source_mutation_is_detected_and_result_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = FakeValidationExecutor(
        source=source,
        mutate_source_on="render_valid",
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    assert "validation.source_asset_modified" in {
        issue.code for issue in run.result.issues
    }
    assert run.result.metadata["source_asset_unchanged"] is False


def test_source_fingerprint_read_error_fails_closed_without_stuck_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    executor = SourceReadFailureExecutor(source=source)

    def fail_late_source_hash(path: str | Path) -> str:
        if executor.block_source_reads and Path(path).resolve() == source:
            raise PermissionError("source became unreadable")
        return file_sha256(path)

    monkeypatch.setattr(
        "content_agent_workflows.validation.workflow.file_sha256",
        fail_late_source_hash,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    assert "validation.source_asset_modified" in {
        issue.code for issue in run.result.issues
    }
    assert all(
        record.state != ValidationWorkItemState.RUNNING
        for record in run.checkpoint.records
    )


def test_source_drift_after_final_commit_cannot_publish_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    source_changed = False

    def change_source_after_completion(checkpoint: object) -> None:
        nonlocal source_changed
        records = getattr(checkpoint, "records")
        if source_changed or not all(
            record.accepted_result is not None for record in records
        ):
            return
        source_changed = True
        source.write_text("changed after final commit\n", encoding="utf-8")

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        progress_callback=change_source_after_completion,
    )

    assert source_changed is True
    assert run.result.verdict == "fail"
    assert run.result.metadata["source_asset_unchanged"] is False
    assert "validation.source_asset_modified" in {
        issue.code for issue in run.result.issues
    }


def test_evidence_drift_after_final_commit_cannot_publish_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_deleted = False

    def delete_evidence_after_completion(checkpoint: object) -> None:
        nonlocal evidence_deleted
        records = getattr(checkpoint, "records")
        if evidence_deleted or not all(
            record.accepted_result is not None for record in records
        ):
            return
        evidence = records[-1].accepted_result.evidence_artifacts
        assert evidence
        Path(evidence[0].path).unlink()
        evidence_deleted = True

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        progress_callback=delete_evidence_after_completion,
    )

    assert evidence_deleted is True
    assert run.result.verdict == "fail"
    assert "validation.accepted_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


def test_reference_drift_after_final_commit_cannot_publish_pass(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    reference_changed = False

    def change_reference_after_completion(checkpoint: object) -> None:
        nonlocal reference_changed
        records = getattr(checkpoint, "records")
        if reference_changed or not all(
            record.accepted_result is not None for record in records
        ):
            return
        _write_image(reference, (1, 2, 3))
        reference_changed = True

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        progress_callback=change_reference_after_completion,
    )

    assert reference_changed is True
    assert run.result.verdict == "fail"
    assert run.result.metadata["reference_evidence_unchanged"] is False
    assert "validation.reference_evidence_stale" in {
        issue.code for issue in run.result.issues
    }


@pytest.mark.parametrize(
    ("artifact_role", "issue_code"),
    (
        ("source", "validation.source_asset_modified"),
        ("reference", "validation.reference_evidence_stale"),
        ("evidence", "validation.accepted_evidence_stale"),
    ),
)
def test_drift_during_final_artifact_publication_cannot_publish_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_role: str,
    issue_code: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    mutated = False

    def mutate_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal mutated
        if not mutated and destination_name == "validation_result.json":
            if artifact_role == "source":
                source.write_text("changed during publication\n", encoding="utf-8")
            elif artifact_role == "reference":
                reference.write_bytes(b"changed during publication\n")
            else:
                (
                    output_dir
                    / "attempts"
                    / "look_right"
                    / "attempt-0001"
                    / "look_right.png"
                ).unlink()
            mutated = True
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        mutate_then_write,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert mutated is True
    assert run.result.verdict == "fail"
    assert issue_code in {issue.code for issue in run.result.issues}
    persisted = json.loads(Path(run.result_path).read_text(encoding="utf-8"))
    assert persisted["verdict"] == "fail"


def test_template_result_symlink_inserted_during_write_is_replaced_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    unrelated = tmp_path / "unrelated-template-result.json"
    unrelated.write_text("do not overwrite\n", encoding="utf-8")
    before = unrelated.read_bytes()
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(workflow_module, "atomic_write_json_at"),
    )
    inserted = False

    def insert_symlink_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal inserted
        destination = output_dir / "attempts" / "render_valid" / "attempt-0001"
        destination /= destination_name
        if not inserted and destination_name == "template_result.json":
            destination.symlink_to(unrelated)
            inserted = True
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        workflow_module,
        "atomic_write_json_at",
        insert_symlink_then_write,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    first_result = (
        output_dir
        / "attempts"
        / "render_valid"
        / "attempt-0001"
        / "template_result.json"
    )
    assert inserted is True
    assert run.result.verdict == "pass"
    assert unrelated.read_bytes() == before
    assert first_result.is_file()
    assert not first_result.is_symlink()


def test_template_result_parent_swap_cannot_redirect_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    attempt_dir = output_dir / "attempts" / "render_valid" / "attempt-0001"
    archived_attempt = tmp_path / "archived-attempt"
    external_attempt = tmp_path / "external-attempt"
    external_attempt.mkdir()
    external_result = external_attempt / "template_result.json"
    external_result.write_text("do not overwrite\n", encoding="utf-8")
    before = external_result.read_bytes()
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(workflow_module, "atomic_write_json_at"),
    )
    swapped = False

    def swap_parent_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal swapped
        if not swapped and destination_name == "template_result.json":
            attempt_dir.rename(archived_attempt)
            attempt_dir.symlink_to(external_attempt, target_is_directory=True)
            swapped = True
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        workflow_module,
        "atomic_write_json_at",
        swap_parent_then_write,
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="attempt directory identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert swapped is True
    assert external_result.read_bytes() == before
    assert (archived_attempt / "template_result.json").is_file()


def test_final_result_symlink_inserted_during_write_is_replaced_safely(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    unrelated = tmp_path / "unrelated-validation-result.json"
    unrelated.write_text("do not overwrite\n", encoding="utf-8")
    before = unrelated.read_bytes()
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    inserted = False

    def insert_symlink_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal inserted
        destination = output_dir / destination_name
        if not inserted and destination_name == "validation_result.json":
            destination.symlink_to(unrelated)
            inserted = True
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        insert_symlink_then_write,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    result_path = output_dir / "validation_result.json"
    assert inserted is True
    assert run.result.verdict == "pass"
    assert unrelated.read_bytes() == before
    assert result_path.is_file()
    assert not result_path.is_symlink()


def test_final_result_parent_swap_cannot_redirect_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    archived_output = tmp_path / "archived-output"
    external_output = tmp_path / "external-output"
    external_output.mkdir()
    external_result = external_output / "validation_result.json"
    external_result.write_text("do not overwrite\n", encoding="utf-8")
    before = external_result.read_bytes()
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    swapped = False

    def swap_parent_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal swapped
        if not swapped and destination_name == "validation_result.json":
            output_dir.rename(archived_output)
            output_dir.symlink_to(external_output, target_is_directory=True)
            swapped = True
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        swap_parent_then_write,
    )

    with pytest.raises(
        RuntimeError,
        match="output directory identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert swapped is True
    assert external_result.read_bytes() == before
    assert not (archived_output / "validation_result.json").exists()


def test_posix_finalizer_requires_precaptured_output_directory_identity(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("Descriptor-pinned finalization is POSIX-specific.")
    placeholder = cast(Any, None)

    with pytest.raises(
        ValueError,
        match="requires a pre-captured output directory identity",
    ):
        finalizer_module.finalize_validation_workflow(
            status=placeholder,
            request=placeholder,
            plan=placeholder,
            raw_result=placeholder,
            checkpoint=placeholder,
            source_before=(),
            source_after=(),
            paths=finalizer_module.ValidationWorkflowPaths.from_output_dir(
                tmp_path / "run"
            ),
            output_dir_identity=None,
        )


def test_failed_bundle_publication_leaves_no_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )

    def fail_evidence_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        if destination_name == "validation_evidence.json":
            raise OSError("injected evidence publication failure")
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        fail_evidence_write,
    )

    with pytest.raises(
        OSError,
        match="injected evidence publication failure",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_checkpoint_replacement_during_final_publication_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    checkpoint_path = output_dir / "validation_checkpoint.json"
    external_checkpoint = tmp_path / "external-checkpoint.json"
    external_before: list[bytes] = []
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    swapped = False

    def swap_checkpoint_after_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal swapped
        original_atomic_write_json_at(parent_fd, destination_name, payload)
        if not swapped and destination_name == "validation_result.json":
            external_checkpoint.write_bytes(checkpoint_path.read_bytes())
            external_before.append(external_checkpoint.read_bytes())
            checkpoint_path.unlink()
            checkpoint_path.symlink_to(external_checkpoint)
            swapped = True

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        swap_checkpoint_after_write,
    )

    with pytest.raises(
        ValidationCheckpointError,
        match="Invalid validation checkpoint",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert swapped is True
    assert checkpoint_path.is_symlink()
    assert external_checkpoint.read_bytes() == external_before[0]
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_bundle_verification_rejects_output_directory_aba_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    archived_output = tmp_path / "archived-output"
    original_verifier = workflow_module.finalized_validation_artifacts_are_valid
    verification_results: list[bool] = []

    def swap_for_first_verification(*args: Any, **kwargs: Any) -> bool:
        if verification_results:
            return original_verifier(*args, **kwargs)
        output_dir.rename(archived_output)
        output_dir.mkdir()
        for artifact_name in (
            "validation_request.json",
            "validation_plan.json",
            "validation_result.json",
            "validation_evidence.json",
            "final_summary.json",
        ):
            (output_dir / artifact_name).write_bytes(
                (archived_output / artifact_name).read_bytes()
            )
        (archived_output / "validation_result.json").write_text(
            '{"verdict": "tampered"}\n',
            encoding="utf-8",
        )
        try:
            verified = original_verifier(*args, **kwargs)
        finally:
            for decoy_artifact in output_dir.iterdir():
                decoy_artifact.unlink()
            output_dir.rmdir()
            archived_output.rename(output_dir)
        verification_results.append(verified)
        return verified

    monkeypatch.setattr(
        workflow_module,
        "finalized_validation_artifacts_are_valid",
        swap_for_first_verification,
    )

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert verification_results[0] is False
    assert run.result.verdict == "pass"
    persisted = json.loads(
        (output_dir / "validation_result.json").read_text(encoding="utf-8")
    )
    assert persisted["verdict"] == "pass"


def test_cross_file_mutation_during_bundle_verification_is_republished(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    request_path = output_dir / "validation_request.json"
    original_json_load = json.load
    mutated = False

    class VerificationJson:
        @staticmethod
        def load(stream: Any) -> Any:
            nonlocal mutated
            payload = original_json_load(stream)
            if not mutated:
                request_path.write_text('{"tampered": true}\n', encoding="utf-8")
                mutated = True
            return payload

    monkeypatch.setattr(finalizer_module, "json", VerificationJson)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    assert mutated is True
    assert run.result.verdict == "pass"
    assert json.loads(request_path.read_text(encoding="utf-8")) == (
        run.request.model_dump(mode="json")
    )


def test_repeated_input_drift_during_publication_leaves_no_terminal_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    original_source = source.read_text(encoding="utf-8")
    changed_source = "changed during alternating publication\n"
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    writes = 0

    def toggle_source_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal writes
        if destination_name == "validation_result.json":
            writes += 1
            source.write_text(
                changed_source if writes % 2 else original_source,
                encoding="utf-8",
            )
        original_atomic_write_json_at(parent_fd, destination_name, payload)

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        toggle_source_then_write,
    )

    with pytest.raises(
        ValidationCheckpointError,
        match="kept changing",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert writes >= 3
    assert not (output_dir / "validation_result.json").exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_repeated_output_tampering_cannot_leave_pass_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    result_path = output_dir / "validation_result.json"
    original_atomic_write_json_at = cast(
        AtomicJsonAtWriter,
        getattr(finalizer_module, "atomic_write_json_at"),
    )
    tampered = 0

    def tamper_prior_result_then_write(
        parent_fd: int,
        destination_name: str,
        payload: Any,
    ) -> None:
        nonlocal tampered
        original_atomic_write_json_at(parent_fd, destination_name, payload)
        if destination_name == "validation_result.json":
            result_path.write_text('{"verdict": "pass"}\n', encoding="utf-8")
            tampered += 1

    monkeypatch.setattr(
        finalizer_module,
        "atomic_write_json_at",
        tamper_prior_result_then_write,
    )

    with pytest.raises(
        ValidationCheckpointError,
        match="kept changing",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert tampered >= 4
    assert not result_path.exists()
    assert not (output_dir / "validation_evidence.json").exists()
    assert not (output_dir / "final_summary.json").exists()


def test_expected_result_policy_cannot_downgrade_final_integrity_failure(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_deleted = False

    def delete_evidence_after_completion(checkpoint: object) -> None:
        nonlocal evidence_deleted
        records = getattr(checkpoint, "records")
        if evidence_deleted or not all(
            record.accepted_result is not None for record in records
        ):
            return
        evidence = records[-1].accepted_result.evidence_artifacts
        assert evidence
        Path(evidence[0].path).unlink()
        evidence_deleted = True

    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["validation.accepted_evidence_stale"],
            },
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        progress_callback=delete_evidence_after_completion,
    )

    assert evidence_deleted is True
    assert run.result.verdict == "fail"
    stale_issue = next(
        issue
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_stale"
    )
    assert stale_issue.severity == "fail"
    assert run.result.metadata["workflow_integrity_expected_result_bypassed"] is True
    assert run.result.request.policy["expected_verdict"] == "fail"
    # The bypass strips the expected-result policy only when computing the
    # verdict. The published request artifact must still match the run's
    # request, so a consumer reading validation_request.json alone sees the
    # policy that was actually configured.
    published_request = json.loads(
        Path(run.request_path).read_text(encoding="utf-8"),
    )
    assert published_request["policy"]["expected_verdict"] == "fail"
    assert published_request["policy"]["expected_issue_codes"] == [
        "validation.accepted_evidence_stale"
    ]
    assert published_request == run.request.model_dump(mode="json")


@pytest.mark.parametrize(
    ("issue_code", "executor_kind"),
    (
        ("validation.inline_credential_rejected", "unsafe_result"),
        ("validation.render_evidence_missing", "missing_render"),
    ),
)
def test_expected_result_policy_cannot_downgrade_workflow_failures(
    tmp_path: Path,
    issue_code: str,
    executor_kind: str,
) -> None:
    source, reference = _inputs(tmp_path)
    secret = "never-persist-this-expected-result-secret"
    executor = (
        UnsafeResultExecutor(source=source, secret=secret)
        if executor_kind == "unsafe_result"
        else MissingRenderEvidenceExecutor(source=source)
    )

    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": [issue_code],
            },
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=executor,
    )

    assert run.result.verdict == "fail"
    issue = next(issue for issue in run.result.issues if issue.code == issue_code)
    assert issue.severity == "fail"
    assert run.result.metadata["workflow_integrity_expected_result_bypassed"] is True
    if executor_kind == "unsafe_result":
        for artifact in Path(run.output_dir).rglob("*.json"):
            assert secret not in artifact.read_text(encoding="utf-8")


def test_warning_code_collision_cannot_hide_final_integrity_failure(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    evidence_deleted = False

    def delete_evidence_after_completion(checkpoint: object) -> None:
        nonlocal evidence_deleted
        records = getattr(checkpoint, "records")
        if evidence_deleted or not all(
            record.accepted_result is not None for record in records
        ):
            return
        evidence = records[-1].accepted_result.evidence_artifacts
        assert evidence
        Path(evidence[0].path).unlink()
        evidence_deleted = True

    run = run_validation_workflow(
        _request(
            source,
            reference,
            policy={
                "expected_verdict": "fail",
                "expected_issue_codes": ["asset.expected_defect"],
            },
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=ReservedIntegrityWarningExecutor(source=source),
        progress_callback=delete_evidence_after_completion,
    )

    assert evidence_deleted is True
    assert run.result.verdict == "fail"
    severities = {
        issue.severity
        for issue in run.result.issues
        if issue.code == "validation.accepted_evidence_stale"
    }
    assert severities == {"warn", "fail"}
    assert run.result.metadata["workflow_integrity_expected_result_bypassed"] is True


def test_custom_checkpoint_store_path_is_published_and_cancellable(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    checkpoint_path = tmp_path / "checkpoint-state" / "custom.json"
    store = FileValidationCheckpointStore(checkpoint_path)

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=store,
    )

    assert run.checkpoint_path == str(checkpoint_path.resolve())
    assert run.plan.artifact_paths["validation_checkpoint"] == run.checkpoint_path
    plan = json.loads(Path(run.plan_path).read_text(encoding="utf-8"))
    assert plan["artifact_paths"]["validation_checkpoint"] == run.checkpoint_path
    assert not (output_dir / "validation_checkpoint.json").exists()

    unchanged = request_validation_cancellation(
        run.output_dir,
        reason="Cancel the custom checkpoint-backed run.",
        checkpoint_store=store,
    )
    assert unchanged.cancellation_requested is False
    assert unchanged == run.checkpoint
    assert store.load() == unchanged


def test_cancellation_rejects_checkpoint_lock_hard_link(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("do not truncate\n", encoding="utf-8")
    before = unrelated.read_bytes()
    checkpoint_lock = output_dir / "validation_checkpoint.json.lock"
    checkpoint_lock.hardlink_to(unrelated)

    with pytest.raises(
        ValidationCheckpointError,
        match="lock path cannot be a hard link",
    ):
        request_validation_cancellation(output_dir)

    assert unrelated.read_bytes() == before
    assert checkpoint_lock.read_bytes() == before


def test_cancellation_recovers_orphaned_plain_checkpoint_lock(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_lock = output_dir / "validation_checkpoint.json.lock"
    checkpoint_lock.write_text("orphaned lock marker\n", encoding="utf-8")

    unchanged = request_validation_cancellation(output_dir)

    assert unchanged == run.checkpoint
    assert not checkpoint_lock.exists()


def test_external_checkpoint_lock_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("do not truncate\n", encoding="utf-8")
    before = unrelated.read_bytes()
    checkpoint_lock = checkpoint_path.with_suffix(".json.lock")
    checkpoint_lock.symlink_to(unrelated)

    with pytest.raises(
        ValidationCheckpointError,
        match="lock path cannot use a symlink",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert unrelated.read_bytes() == before
    assert checkpoint_lock.is_symlink()


def test_mid_run_checkpoint_symlink_swap_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    checkpoint_path = output_dir / "validation_checkpoint.json"
    external_checkpoint = tmp_path / "external-checkpoint.json"
    external_before: list[bytes] = []
    swapped = False

    def swap_checkpoint(checkpoint: ValidationWorkflowCheckpoint) -> None:
        nonlocal swapped
        look_right = next(
            record
            for record in checkpoint.records
            if record.template_name == "look_right"
        )
        if swapped or look_right.state != ValidationWorkItemState.RUNNING:
            return
        external_checkpoint.write_bytes(checkpoint_path.read_bytes())
        external_before.append(external_checkpoint.read_bytes())
        checkpoint_path.unlink()
        checkpoint_path.symlink_to(external_checkpoint)
        swapped = True

    with pytest.raises(
        ValidationCheckpointError,
        match="Invalid validation checkpoint",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            progress_callback=swap_checkpoint,
        )

    assert swapped is True
    assert checkpoint_path.is_symlink()
    assert external_checkpoint.read_bytes() == external_before[0]


def test_mid_run_checkpoint_parent_symlink_swap_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    original_state_dir = tmp_path / "original-state"
    external_state_dir = tmp_path / "external-state"
    external_checkpoint = external_state_dir / checkpoint_path.name
    external_before: list[bytes] = []
    swapped = False

    def swap_checkpoint_parent(checkpoint: ValidationWorkflowCheckpoint) -> None:
        nonlocal swapped
        look_right = next(
            record
            for record in checkpoint.records
            if record.template_name == "look_right"
        )
        if swapped or look_right.state != ValidationWorkItemState.RUNNING:
            return
        state_dir.rename(original_state_dir)
        external_state_dir.mkdir()
        external_checkpoint.write_bytes(
            (original_state_dir / checkpoint_path.name).read_bytes()
        )
        external_before.append(external_checkpoint.read_bytes())
        state_dir.symlink_to(external_state_dir, target_is_directory=True)
        swapped = True

    with pytest.raises(
        ValidationCheckpointError,
        match="parent directory could not be opened safely",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
            progress_callback=swap_checkpoint_parent,
        )

    assert swapped is True
    assert state_dir.is_symlink()
    assert external_checkpoint.read_bytes() == external_before[0]


def test_checkpoint_ancestor_swap_before_first_operation_is_rejected(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    outer = tmp_path / "outer"
    state_dir = outer / "state"
    state_dir.mkdir(parents=True)
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    original_outer = tmp_path / "original-outer"
    external_outer = tmp_path / "external-outer"
    external_state = external_outer / "state"
    external_state.mkdir(parents=True)
    external_checkpoint = external_state / checkpoint_path.name
    external_checkpoint.write_text("do not overwrite\n", encoding="utf-8")
    before = external_checkpoint.read_bytes()
    outer.rename(original_outer)
    outer.symlink_to(external_outer, target_is_directory=True)

    with pytest.raises(
        ValidationCheckpointError,
        match="parent directory",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert outer.is_symlink()
    assert external_checkpoint.read_bytes() == before
    assert not (original_outer / "state" / checkpoint_path.name).exists()


@pytest.mark.parametrize("swap_target", ("temporary", "destination"))
def test_checkpoint_entry_swap_during_atomic_replace_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    swap_target: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    unrelated = tmp_path / "unrelated-checkpoint.json"
    unrelated.write_text("do not overwrite\n", encoding="utf-8")
    before = unrelated.read_bytes()
    original_replace = os.replace
    swapped = False

    def swap_checkpoint_entry(
        src: Any,
        dst: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        src_dir_fd = kwargs.get("src_dir_fd")
        dst_dir_fd = kwargs.get("dst_dir_fd")
        is_checkpoint_replace = (
            not swapped
            and dst == checkpoint_path.name
            and src_dir_fd is not None
            and dst_dir_fd is not None
        )
        if is_checkpoint_replace and swap_target == "temporary":
            os.unlink(src, dir_fd=src_dir_fd)
            os.symlink(unrelated, src, dir_fd=src_dir_fd)
            swapped = True
        original_replace(src, dst, *args, **kwargs)
        if is_checkpoint_replace and swap_target == "destination":
            os.unlink(dst, dir_fd=dst_dir_fd)
            os.symlink(unrelated, dst, dir_fd=dst_dir_fd)
            swapped = True

    monkeypatch.setattr(os, "replace", swap_checkpoint_entry)

    with pytest.raises(
        ValidationCheckpointError,
        match="destination identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert swapped is True
    assert unrelated.read_bytes() == before
    assert checkpoint_path.is_symlink()


def test_checkpoint_parent_swap_during_atomic_replace_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    archived_state = tmp_path / "archived-state"
    original_replace = os.replace
    swapped = False

    def swap_checkpoint_parent(
        src: Any,
        dst: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and dst == checkpoint_path.name
            and kwargs.get("dst_dir_fd") is not None
        ):
            state_dir.rename(archived_state)
            state_dir.mkdir()
            swapped = True
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_checkpoint_parent)

    with pytest.raises(
        ValidationCheckpointError,
        match="parent directory identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert swapped is True
    assert not checkpoint_path.exists()
    assert (archived_state / checkpoint_path.name).is_file()


def test_checkpoint_lock_swap_while_held_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    checkpoint_path = state_dir / "validation.json"
    checkpoint_lock = checkpoint_path.with_suffix(".json.lock")
    displaced_lock = checkpoint_lock.with_suffix(".lock.displaced")
    store = FileValidationCheckpointStore(checkpoint_path)
    original_write = store._atomic_write_unlocked  # noqa: SLF001
    swapped = False

    def swap_lock_then_write(
        checkpoint: ValidationWorkflowCheckpoint,
        *,
        expected_identity: tuple[int, int] | None,
    ) -> None:
        nonlocal swapped
        checkpoint_lock.rename(displaced_lock)
        checkpoint_lock.write_text("replacement lock\n", encoding="utf-8")
        swapped = True
        original_write(checkpoint, expected_identity=expected_identity)

    monkeypatch.setattr(store, "_atomic_write_unlocked", swap_lock_then_write)

    with pytest.raises(
        ValidationCheckpointError,
        match="lock path identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert swapped is True
    assert checkpoint_lock.read_text(encoding="utf-8") == "replacement lock\n"
    assert displaced_lock.is_file()


def test_checkpoint_intermediate_ancestor_symlink_back_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    outer = tmp_path / "outer"
    state_dir = outer / "state"
    state_dir.mkdir(parents=True)
    checkpoint_path = state_dir / "validation.json"
    store = FileValidationCheckpointStore(checkpoint_path)
    archived_outer = tmp_path / "archived-outer"
    original_replace = os.replace
    swapped = False

    def swap_ancestor_back_to_same_parent(
        src: Any,
        dst: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        if (
            not swapped
            and dst == checkpoint_path.name
            and kwargs.get("dst_dir_fd") is not None
        ):
            outer.rename(archived_outer)
            outer.symlink_to(archived_outer, target_is_directory=True)
            swapped = True
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_ancestor_back_to_same_parent)

    with pytest.raises(
        ValidationCheckpointError,
        match="parent directory identity changed",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert swapped is True
    assert outer.is_symlink()
    assert (archived_outer / "state" / checkpoint_path.name).is_file()


def test_artifact_write_intermediate_ancestor_symlink_back_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    outer = tmp_path / "outer"
    output_dir = outer / "run"
    outer.mkdir()
    archived_outer = tmp_path / "archived-outer"
    original_replace = os.replace
    swapped = False

    def swap_ancestor_back_to_same_parent(
        src: Any,
        dst: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        nonlocal swapped
        if not swapped and dst == "validation_request.json":
            outer.rename(archived_outer)
            outer.symlink_to(archived_outer, target_is_directory=True)
            swapped = True
        original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_ancestor_back_to_same_parent)

    with pytest.raises(
        RuntimeError,
        match="parent directory changed during write",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert swapped is True
    assert outer.is_symlink()
    assert (archived_outer / "run" / "validation_request.json").is_file()


def test_checkpoint_lock_replacement_during_acquisition_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    checkpoint_path = state_dir / "validation.json"
    checkpoint_lock = checkpoint_path.with_suffix(".json.lock")
    displaced_lock = checkpoint_lock.with_suffix(".lock.displaced")
    store = FileValidationCheckpointStore(checkpoint_path)
    original_open = os.open
    swapped = False

    def replace_lock_after_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if not swapped and path == checkpoint_lock.name and dir_fd is not None:
            os.rename(
                checkpoint_lock.name,
                displaced_lock.name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            replacement_fd = original_open(
                checkpoint_lock.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=dir_fd,
            )
            os.close(replacement_fd)
            swapped = True
        return fd

    monkeypatch.setattr(os, "open", replace_lock_after_open)

    with pytest.raises(
        ValidationCheckpointError,
        match="identity changed during acquisition",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert swapped is True
    assert checkpoint_lock.is_file()
    assert displaced_lock.is_file()


def test_atomic_write_parent_swap_does_not_leak_into_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    archived_parent = tmp_path / "archived-parent"
    destination = parent / "result.json"
    original_create = cast(
        Callable[..., tuple[int, str]],
        getattr(artifacts_module, "_create_temporary_file_at"),
    )
    swapped = False

    def swap_then_create(
        parent_fd: int,
        *,
        destination_name: str,
    ) -> tuple[int, str]:
        nonlocal swapped
        parent.rename(archived_parent)
        parent.mkdir()
        swapped = True
        return original_create(
            parent_fd,
            destination_name=destination_name,
        )

    monkeypatch.setattr(
        artifacts_module,
        "_create_temporary_file_at",
        swap_then_create,
    )
    atomic_write_json = cast(
        AtomicJsonWriter,
        getattr(artifacts_module, "atomic_write_json"),
    )

    with pytest.raises(RuntimeError, match="parent directory changed"):
        atomic_write_json(destination, {"payload": "sensitive"})

    assert swapped is True
    assert not any(parent.iterdir())
    archived_result = archived_parent / destination.name
    assert json.loads(archived_result.read_text(encoding="utf-8")) == {
        "payload": "sensitive"
    }
    assert not any(
        path.name.startswith(f".{destination.name}.")
        for path in archived_parent.iterdir()
    )


def test_atomic_writes_and_checkpoints_accept_existing_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    atomic_write_json = cast(
        AtomicJsonWriter,
        getattr(artifacts_module, "atomic_write_json"),
    )
    artifact_path = linked_parent / "artifacts" / "result.json"

    written_path = atomic_write_json(artifact_path, {"status": "ok"})
    store = FileValidationCheckpointStore(linked_parent / "state" / "validation.json")

    assert written_path == real_parent / "artifacts" / "result.json"
    assert json.loads(written_path.read_text(encoding="utf-8")) == {"status": "ok"}
    assert store.path == real_parent / "state" / "validation.json"


def test_custom_checkpoint_store_ignores_unused_default_checkpoint_kind(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    unused_default_checkpoint = output_dir / "validation_checkpoint.json"
    unused_default_checkpoint.mkdir(parents=True)
    store = FileValidationCheckpointStore(tmp_path / "custom-checkpoint.json")

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=store,
    )

    assert run.result.verdict == "pass"
    assert Path(run.checkpoint_path) == store.path
    assert store.path.is_file()
    assert unused_default_checkpoint.is_dir()


def test_distinct_checkpoint_stores_cannot_share_one_output_directory(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"
    first_store = FileValidationCheckpointStore(tmp_path / "state-1.json")
    second_store = FileValidationCheckpointStore(tmp_path / "state-2.json")
    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=first_store,
    )
    before = Path(first.result_path).read_bytes()

    with pytest.raises(
        ValidationWorkflowError,
        match="owned by a different checkpoint store",
    ):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=second_store,
        )

    assert Path(first.result_path).read_bytes() == before
    assert not second_store.path.exists()


def test_output_owner_symlink_is_rejected_without_following_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    store = FileValidationCheckpointStore(tmp_path / "state.json")
    owner_target = tmp_path / "owner-target.json"
    owner_target.write_text(
        json.dumps(
            {
                "schema_version": "content-agent-workflows.validation-run-owner.v1",
                "checkpoint_path": str(store.path),
            }
        ),
        encoding="utf-8",
    )
    before = owner_target.read_bytes()
    owner_path = output_dir / ".validation_workflow_owner.json"
    original_validate_checkpoint_location = (
        workflow_module._validate_checkpoint_location
    )

    def insert_owner_after_validation(
        checkpoint_path: Path,
        identity: Any,
    ) -> None:
        original_validate_checkpoint_location(checkpoint_path, identity)
        owner_path.symlink_to(owner_target)

    monkeypatch.setattr(
        workflow_module,
        "_validate_checkpoint_location",
        insert_owner_after_validation,
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="owner marker is invalid",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert owner_target.read_bytes() == before
    assert not store.path.exists()


def test_output_owner_fifo_is_rejected_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    store = FileValidationCheckpointStore(tmp_path / "state.json")
    owner_path = output_dir / ".validation_workflow_owner.json"
    original_validate_checkpoint_location = (
        workflow_module._validate_checkpoint_location
    )

    def insert_owner_after_validation(
        checkpoint_path: Path,
        identity: Any,
    ) -> None:
        original_validate_checkpoint_location(checkpoint_path, identity)
        os.mkfifo(owner_path)

    monkeypatch.setattr(
        workflow_module,
        "_validate_checkpoint_location",
        insert_owner_after_validation,
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="owned by a different checkpoint store",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not store.path.exists()


def test_output_owner_is_checked_before_planner_can_mutate_existing_run(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"
    first_store = FileValidationCheckpointStore(tmp_path / "state-1.json")
    second_store = FileValidationCheckpointStore(tmp_path / "state-2.json")
    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
        checkpoint_store=first_store,
    )
    result_path = Path(first.result_path)
    before = result_path.read_bytes()
    planner = PlannerOutputMutatingExecutor(
        source=source,
        output_path=result_path,
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="owned by a different checkpoint store",
    ):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=planner,
            checkpoint_store=second_store,
        )

    assert planner.plan_calls == 0
    assert result_path.read_bytes() == before


def test_custom_checkpoint_store_inside_source_directory_is_rejected(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_source(source_dir / "toolbox.usda")
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    checkpoint_path = source_dir / "validation-checkpoint.json"
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store.path cannot be inside a source directory",
    ):
        run_validation_workflow(
            _request(source_dir, reference),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".json.lock").exists()


def test_custom_checkpoint_store_inside_reference_directory_is_rejected(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "toolbox.usda")
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    _write_image(reference_dir / "reference.png", (220, 180, 0))
    checkpoint_path = reference_dir / "validation-checkpoint.json"
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store.path cannot be inside a reference directory",
    ):
        run_validation_workflow(
            _request(source, reference_dir),
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".json.lock").exists()


@pytest.mark.parametrize(
    "artifact_name",
    (
        "validation_request.json",
        "validation_plan.json",
        "validation_result.json",
        "validation_evidence.json",
        "final_summary.json",
        "attempts",
        "attempts/render_valid/attempt-0001/template_result.json",
    ),
)
def test_custom_checkpoint_store_cannot_alias_workflow_artifacts(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    checkpoint_path = output_dir / artifact_name
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store.path cannot alias",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.lock").exists()


@pytest.mark.parametrize(
    "artifact_name",
    (
        "validation_request.json",
        "validation_plan.json",
        "validation_result.json",
        "validation_evidence.json",
        "final_summary.json",
    ),
)
def test_custom_checkpoint_store_cannot_descend_from_workflow_file_artifacts(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    checkpoint_path = output_dir / artifact_name / "checkpoint.json"
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store.path cannot alias or be inside",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".json.lock").exists()
    assert not output_dir.exists()


def test_custom_checkpoint_store_cannot_contain_output_directory(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    checkpoint_path = tmp_path / "state.json"
    output_dir = checkpoint_path / "run"
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store.path cannot contain",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".json.lock").exists()
    assert not output_dir.exists()


@pytest.mark.parametrize("nested", (False, True))
def test_custom_checkpoint_lock_cannot_contain_output_directory(
    tmp_path: Path,
    nested: bool,
) -> None:
    source, reference = _inputs(tmp_path)
    checkpoint_path = tmp_path / "state.json"
    checkpoint_lock = checkpoint_path.with_suffix(".json.lock")
    output_dir = checkpoint_lock / "run" if nested else checkpoint_lock
    store = FileValidationCheckpointStore(checkpoint_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="checkpoint_store lock path cannot contain",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert not checkpoint_path.exists()
    assert not checkpoint_lock.exists()
    assert not output_dir.exists()


@pytest.mark.parametrize("input_role", ("source", "reference"))
@pytest.mark.parametrize(
    "artifact_name",
    (
        "validation_request.json",
        "validation_plan.json",
        "validation_result.json",
        "validation_checkpoint.json",
        "validation_checkpoint.json.lock",
        "validation_evidence.json",
        "final_summary.json",
        "attempts/render_valid/attempt-0001/template_result.json",
    ),
)
def test_input_artifacts_cannot_alias_workflow_outputs(
    tmp_path: Path,
    input_role: str,
    artifact_name: str,
) -> None:
    output_dir = tmp_path / "run"
    collision_path = output_dir / artifact_name
    collision_path.parent.mkdir(parents=True, exist_ok=True)
    if input_role == "source":
        source = _write_source(collision_path)
        reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    else:
        source = _write_source(tmp_path / "toolbox.usda")
        collision_path.write_bytes(b"reference-input")
        reference = collision_path.resolve()
    before = collision_path.read_bytes()

    with pytest.raises(
        ValidationWorkflowError,
        match="input artifact cannot alias",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert collision_path.read_bytes() == before
    assert not any(
        path.exists()
        for path in (
            output_dir / "validation_request.json",
            output_dir / "validation_plan.json",
            output_dir / "validation_result.json",
            output_dir / "validation_checkpoint.json",
            output_dir / "validation_evidence.json",
            output_dir / "final_summary.json",
        )
        if path != collision_path
    )


@pytest.mark.parametrize("input_role", ("source", "reference"))
def test_checkpoint_lock_hard_link_cannot_alias_input_artifact(
    tmp_path: Path,
    input_role: str,
) -> None:
    source = _write_source(tmp_path / "toolbox.usda")
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    input_path = source if input_role == "source" else reference
    before = input_path.read_bytes()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint_lock = output_dir / "validation_checkpoint.json.lock"
    checkpoint_lock.hardlink_to(input_path)

    with pytest.raises(
        ValidationWorkflowError,
        match="lock path cannot be a hard link",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert input_path.read_bytes() == before
    assert checkpoint_lock.read_bytes() == before
    assert not (output_dir / "validation_checkpoint.json").exists()


def test_checkpoint_lock_hard_link_cannot_alias_unrelated_file(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("do not truncate\n", encoding="utf-8")
    before = unrelated.read_bytes()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint_lock = output_dir / "validation_checkpoint.json.lock"
    checkpoint_lock.hardlink_to(unrelated)

    with pytest.raises(
        ValidationWorkflowError,
        match="lock path cannot be a hard link",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert unrelated.read_bytes() == before
    assert checkpoint_lock.read_bytes() == before
    assert not (output_dir / "validation_checkpoint.json").exists()


@pytest.mark.parametrize(
    "artifact_name",
    (
        "validation_checkpoint.json",
        "validation_result.json",
        "attempts/look_right/attempt-0001/template_result.json",
    ),
)
def test_checkpoint_lock_hard_link_cannot_alias_workflow_artifact(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    output_dir = tmp_path / "run"
    first = run_validation_workflow(
        request,
        output_dir=output_dir,
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(first.checkpoint_path)
    checkpoint_lock = checkpoint_path.with_suffix(".json.lock")
    artifact_path = output_dir / artifact_name
    checkpoint_lock.unlink(missing_ok=True)
    checkpoint_lock.hardlink_to(artifact_path)
    before = artifact_path.read_bytes()

    with pytest.raises(
        ValidationWorkflowError,
        match="lock path cannot be a hard link",
    ):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            resume=True,
        )

    assert artifact_path.read_bytes() == before
    assert checkpoint_lock.read_bytes() == before


@pytest.mark.parametrize("input_role", ("source", "reference"))
def test_checkpoint_lock_hard_link_cannot_alias_input_directory_member(
    tmp_path: Path,
    input_role: str,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_source(source_dir / "toolbox.usda")
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = _write_image(reference_dir / "reference.png", (220, 180, 0))
    input_path = source if input_role == "source" else reference
    before = input_path.read_bytes()
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    checkpoint_lock = output_dir / "validation_checkpoint.json.lock"
    checkpoint_lock.hardlink_to(input_path)
    request = _request(source_dir, reference)
    if input_role == "reference":
        request = _request(source, reference_dir)

    with pytest.raises(
        ValidationWorkflowError,
        match="lock path cannot be a hard link",
    ):
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert input_path.read_bytes() == before
    assert checkpoint_lock.read_bytes() == before
    assert not (output_dir / "validation_checkpoint.json").exists()


def test_existing_run_requires_explicit_resume(tmp_path: Path) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference)
    run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    with pytest.raises(ValidationCheckpointError, match="resume=True"):
        run_validation_workflow(
            request,
            output_dir=tmp_path / "run",
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )


def test_missing_checkpoint_load_does_not_create_parent_directories(
    tmp_path: Path,
) -> None:
    checkpoint_parent = tmp_path / "missing" / "nested"
    store = FileValidationCheckpointStore(
        checkpoint_parent / "validation_checkpoint.json"
    )

    assert store.load() is None
    assert not checkpoint_parent.exists()


def test_resume_without_checkpoint_does_not_claim_output_directory(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "missing-run"

    with pytest.raises(
        ValidationCheckpointError,
        match="Cannot resume because no validation checkpoint exists",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            resume=True,
        )

    assert not output_dir.exists()


def test_non_linux_posix_platform_is_rejected_before_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    monkeypatch.setattr(workflow_module.sys, "platform", "darwin")

    with pytest.raises(
        ValidationWorkflowError,
        match="requires Linux, a Linux container, or WSL2",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert not output_dir.exists()


def test_inline_request_credential_is_rejected_before_artifacts(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    secret = "never-persist-this-validation-secret"
    request = _request(source, reference).model_copy(
        update={
            "policy": {
                **_request(source, reference).policy,
                "look_right_vlm": {"api_key": secret},
            }
        }
    )
    output_dir = tmp_path / "run"

    with pytest.raises(InlineSecretError) as exc_info:
        run_validation_workflow(
            request,
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert secret not in str(exc_info.value)
    assert not output_dir.exists()


def test_unsafe_executor_result_becomes_safe_structured_failure(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    secret = "never-persist-this-executor-secret"

    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=UnsafeResultExecutor(source=source, secret=secret),
    )

    assert run.result.verdict == "fail"
    assert "validation.inline_credential_rejected" in {
        issue.code for issue in run.result.issues
    }
    for artifact in Path(run.output_dir).rglob("*.json"):
        assert secret not in artifact.read_text(encoding="utf-8")


def test_explicit_env_reference_and_benign_token_fields_are_persisted(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(source, reference).model_copy(
        update={
            "policy": {
                **_request(source, reference).policy,
                "look_right_vlm": {
                    "api_key_env": "${NVIDIA_API_KEY}",
                    "max_tokens": 256,
                    "tokenizer": "default",
                },
            }
        }
    )

    run = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )

    persisted = json.loads(Path(run.request_path).read_text(encoding="utf-8"))
    assert persisted["policy"]["look_right_vlm"] == {
        "api_key_env": "${NVIDIA_API_KEY}",
        "max_tokens": 256,
        "tokenizer": "default",
    }


def test_inline_credential_in_cancellation_reason_is_not_persisted(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(run.checkpoint_path)
    before = checkpoint_path.read_text(encoding="utf-8")
    secret = "never-persist-this-cancellation-secret"

    with pytest.raises(InlineSecretError) as exc_info:
        request_validation_cancellation(
            run.output_dir,
            reason=f"Authorization: Bearer {secret}",
        )

    assert secret not in str(exc_info.value)
    assert checkpoint_path.read_text(encoding="utf-8") == before


def test_checkpoint_load_rejects_tampered_inline_credential(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(run.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    secret = "never-trust-this-tampered-checkpoint-secret"
    payload["cancellation_requested"] = True
    payload["cancellation_reason"] = f"Authorization: Bearer {secret}"
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationCheckpointError) as exc_info:
        FileValidationCheckpointStore(checkpoint_path).load()

    assert secret not in str(exc_info.value)


@pytest.mark.parametrize(
    "raw_template",
    [
        "https://assets.example.test/state.json?X-Amz-Signature={secret}",
        "https://{secret}@assets.example.test/state.json",
    ],
    ids=["signed-query", "username-only-userinfo"],
)
def test_checkpoint_path_inline_secret_is_rejected_before_artifacts(
    tmp_path: Path,
    raw_template: str,
) -> None:
    source, reference = _inputs(tmp_path)
    secret = "checkpoint-path-secret-713"
    checkpoint_path = (tmp_path / Path(raw_template.format(secret=secret))).resolve()
    store = FileValidationCheckpointStore(checkpoint_path)
    output_dir = tmp_path / "run"

    with pytest.raises(InlineSecretError) as exc_info:
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
            checkpoint_store=store,
        )

    assert secret not in str(exc_info.value)
    assert not output_dir.exists()
    assert not store.path.exists()
    assert not store.path.with_suffix(f"{store.path.suffix}.lock").exists()


@pytest.mark.parametrize("schema_owner", ("checkpoint", "workflow_identity"))
def test_checkpoint_load_rejects_unsupported_schema_versions(
    tmp_path: Path,
    schema_owner: str,
) -> None:
    source, reference = _inputs(tmp_path)
    run = run_validation_workflow(
        _request(source, reference),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=FakeValidationExecutor(source=source),
    )
    checkpoint_path = Path(run.checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if schema_owner == "checkpoint":
        payload["schema_version"] = "content-agent-workflows.validation-checkpoint.v2"
    else:
        payload["workflow_identity"]["schema_version"] = (
            "content-agent-workflows.validation-identity.v2"
        )
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValidationCheckpointError,
        match="Invalid validation checkpoint",
    ):
        FileValidationCheckpointStore(checkpoint_path).load()


def test_output_directory_inside_source_directory_is_rejected(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = _write_source(source_dir / "toolbox.usda")
    reference = _write_image(tmp_path / "reference.png", (220, 180, 0))
    output_dir = source_dir / "validation-run"

    with pytest.raises(ValidationWorkflowError, match="inside a source directory"):
        run_validation_workflow(
            _request(source_dir, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert not output_dir.exists()


def test_output_directory_inside_reference_directory_is_rejected(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path / "toolbox.usda")
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    _write_image(reference_dir / "reference.png", (220, 180, 0))
    output_dir = reference_dir / "validation-run"

    with pytest.raises(
        ValidationWorkflowError,
        match="inside a reference directory",
    ):
        run_validation_workflow(
            _request(source, reference_dir),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "artifact_name",
    (
        "validation_request.json",
        "validation_plan.json",
        "validation_result.json",
        "validation_checkpoint.json",
        "validation_checkpoint.json.lock",
        "validation_evidence.json",
        "final_summary.json",
        "attempts",
    ),
)
def test_preexisting_canonical_artifact_wrong_type_is_rejected_before_execution(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    collision_path = output_dir / artifact_name
    if artifact_name == "attempts":
        collision_path.write_text("not a directory\n", encoding="utf-8")
    else:
        collision_path.mkdir()
    executor = FakeValidationExecutor(source=source)

    with pytest.raises(
        ValidationWorkflowError,
        match="workflow artifact must be a (file|directory)",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.calls == []


def test_preexisting_canonical_non_regular_file_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    os.mkfifo(output_dir / "validation_result.json")
    executor = FakeValidationExecutor(source=source)

    with pytest.raises(
        ValidationWorkflowError,
        match="validation_result workflow artifact must be a file",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=executor,
        )

    assert executor.plan_calls == 0
    assert executor.calls == []


@pytest.mark.parametrize(
    ("artifact_name", "is_directory"),
    (
        ("validation_request.json", False),
        ("validation_plan.json", False),
        ("validation_result.json", False),
        ("validation_checkpoint.json", False),
        ("validation_checkpoint.json.lock", False),
        ("validation_evidence.json", False),
        ("final_summary.json", False),
        ("attempts", True),
    ),
)
def test_preexisting_canonical_artifact_symlink_is_rejected(
    tmp_path: Path,
    artifact_name: str,
    is_directory: bool,
) -> None:
    source, reference = _inputs(tmp_path)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    target = tmp_path / f"outside-{artifact_name.replace('.', '-')}"
    if is_directory:
        target.mkdir()
    else:
        target.write_text("sentinel\n", encoding="utf-8")
    (output_dir / artifact_name).symlink_to(
        target,
        target_is_directory=is_directory,
    )

    with pytest.raises(
        ValidationWorkflowError,
        match="workflow artifact cannot use a symlink",
    ):
        run_validation_workflow(
            _request(source, reference),
            output_dir=output_dir,
            config_base_dir=tmp_path,
            executor=FakeValidationExecutor(source=source),
        )

    if is_directory:
        assert list(target.iterdir()) == []
    else:
        assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_scaffold_step_executor_rehydrates_render_handoff_without_rerendering(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    render = _write_image(tmp_path / "accepted-render.png", (230, 180, 0))
    request = _request(
        source,
        reference,
        policy={
            "look_right_response": """
Critique: The render matches the yellow toolbox reference.
Score: 9
Decision: PASS
Issue Codes: none
"""
        },
    ).model_copy(update={"requested_templates": ("render_valid", "look_right")})
    executor = ScaffoldValidationStepExecutor(tmp_path)
    plan = executor.plan(request, working_dir=tmp_path / "run")
    render_result = ValidationTemplateResult(
        template_name="render_valid",
        status="passed",
        metadata={
            "runtime_render": {
                "status": "passed",
                "backend": "checkpoint",
                "image_paths": [str(render)],
                "render_response": None,
                "render_output_dir": str(tmp_path),
                "issues": [],
                "metadata": {},
            },
            "adapter_result": {
                "status": "pass",
                "verdict": "pass",
                "issues": [],
            },
        },
    )

    result = executor.run(
        "look_right",
        ValidationTemplateContext(
            request=request,
            plan=plan,
            working_dir=tmp_path / "look-attempt",
            previous_template_results=(render_result,),
        ),
    )

    assert result.status == "passed"
    assert result.metrics["vlm_invoked"] is False
    assert result.evidence["render_valid_handoff"]["status"] == "pass"


def test_workflow_preserves_relative_render_handoff_path_base(
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    request = _request(
        source,
        reference,
        policy={
            "look_right_response": """
Critique: The render matches the yellow toolbox reference.
Score: 9
Decision: PASS
Issue Codes: none
"""
        },
    )

    run = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=RelativeRenderHandoffExecutor(config_base_dir=tmp_path),
    )

    assert run.result.verdict == "pass"
    render_record = run.checkpoint.records[0]
    assert render_record.accepted_result is not None
    runtime_render = render_record.accepted_result.result.metadata["runtime_render"]
    expected_render = (
        tmp_path / "run" / "attempts" / "render_valid" / "attempt-0001" / "render.png"
    )
    assert runtime_render["image_paths"] == [str(expected_render)]
    assert runtime_render["render_response"]["results"][0]["images"] == [
        str(expected_render)
    ]


def test_scaffold_renderer_source_metadata_is_not_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        del policy
        render_dir = Path(working_dir) / "renders" / "toolbox"
        render_dir.mkdir(parents=True)
        render_path = render_dir / "toolbox_corner_0000.png"
        image = Image.new("RGB", (64, 64), color=(220, 180, 0))
        for offset in range(64):
            image.putpixel((offset, offset), (20, 20, 20))
            image.putpixel((63 - offset, offset), (80, 70, 10))
        image.save(render_path)
        return {
            "status": "completed",
            "backend": "fake",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "fake",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "usd_path": str(usd_paths[0]),
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1, "views": ["corner"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )

    run = run_validation_workflow(
        _request(
            source,
            reference,
            requested_templates=("render_valid",),
        ),
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        executor=ScaffoldValidationStepExecutor(tmp_path),
    )

    assert run.result.verdict == "pass"
    accepted = run.checkpoint.records[0].accepted_result
    assert accepted is not None
    evidence_paths = {artifact.path for artifact in accepted.evidence_artifacts}
    assert str(source) not in evidence_paths
    assert any(path.endswith("toolbox_corner_0000.png") for path in evidence_paths)


def test_scaffold_workflow_resumes_toolbox_prompt_without_rerendering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, reference = _inputs(tmp_path)
    render_calls: list[tuple[str, ...]] = []

    def fake_runtime_renderer(
        *,
        usd_paths: Sequence[str | Path],
        working_dir: str | Path,
        policy: Mapping[str, Any],
    ) -> dict[str, object]:
        del policy
        render_calls.append(tuple(str(path) for path in usd_paths))
        render_dir = Path(working_dir) / "renders" / "toolbox"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_path = render_dir / "toolbox_corner_0000.png"
        image = Image.new("RGB", (64, 64), color=(220, 180, 0))
        for offset in range(64):
            image.putpixel((offset, offset), (20, 20, 20))
            image.putpixel((63 - offset, offset), (80, 70, 10))
        image.save(render_path)
        return {
            "status": "completed",
            "backend": "fake",
            "image_paths": [str(render_path)],
            "render_response": {
                "backend": "fake",
                "status": "completed",
                "results": [
                    {
                        "camera": "corner",
                        "camera_path": "/ValidationAgentCameras/corner",
                        "images": [str(render_path)],
                        "frame_count": 1,
                        "status": "success",
                    }
                ],
            },
            "render_output_dir": str(render_dir),
            "issues": [],
            "metadata": {"image_count": 1, "views": ["corner"]},
        }

    monkeypatch.setattr(
        "world_understanding.agentic.validation_scaffold.render_usd_visual_evidence",
        fake_runtime_renderer,
    )
    request = _request(source, reference).model_copy(
        update={
            "policy": {
                **_request(source, reference).policy,
                "look_right_response": """
Critique: The current yellow toolbox render matches the supplied reference.
Score: 9
Decision: PASS
Issue Codes: none
""",
            }
        }
    )
    token = ValidationCancellationToken()

    def cancel_after_render(checkpoint: object) -> None:
        records = getattr(checkpoint, "records")
        if records[0].accepted_result is not None:
            token.cancel()

    first = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        cancellation_token=token,
        progress_callback=cancel_after_render,
    )
    resumed = run_validation_workflow(
        request,
        output_dir=tmp_path / "run",
        config_base_dir=tmp_path,
        resume=True,
    )

    assert first.status.value == "cancelled"
    assert resumed.status.value == "completed"
    assert resumed.result.verdict == "pass"
    assert render_calls == [(str(source),)]


def test_plural_artifact_marker_precedence_is_behaviourally_neutral() -> None:
    """Pin why ``A or (B and C)`` precedence is safe in the marker check.

    ``_has_plural_artifact_path_record_marker`` reads as
    ``exact_name or (suffix and predicate)``, so the exact names skip the
    predicate. That is only harmless because every exact name also satisfies
    both predicates in use. If a predicate ever stops matching one of them the
    precedence would start to matter, so assert the invariant directly.
    """

    for predicate in (
        workflow_module._is_artifact_path_key,
        workflow_module._is_typed_evidence_artifact_path_key,
    ):
        for exact_key in ("images", "paths", "uris"):
            assert predicate(exact_key) is True
            assert workflow_module._has_plural_artifact_path_record_marker(
                {exact_key: ["evidence.png"]},
                path_key_predicate=predicate,
            )

    # A suffixed name is ambiguous, so it must also satisfy the predicate.
    assert workflow_module._has_plural_artifact_path_record_marker(
        {"image_paths": ["evidence.png"]},
        path_key_predicate=workflow_module._is_typed_evidence_artifact_path_key,
    )
    assert not workflow_module._has_plural_artifact_path_record_marker(
        {"scene_paths": ["/World/Toolbox"]},
        path_key_predicate=lambda _key: False,
    )
