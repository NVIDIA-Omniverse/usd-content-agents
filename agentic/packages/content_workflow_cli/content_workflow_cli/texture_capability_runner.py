# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI adapters for independently selected Texture capabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from content_agent_workflows.texture import (
        TextureAgentServiceClient,
        TextureGeneratorInputs,
        TextureGeneratorLeafRequest,
    )


def _texture_runtime() -> Any:
    """Load Texture implementation only after a focused command is selected."""

    import content_agent_workflows.texture as texture

    return texture


def _live_usd_cli_texture_validator(**kwargs: Any) -> Any:
    return _texture_runtime().LiveUsdCliTextureValidator(**kwargs)


# Preserve the existing test/integration patch seam without importing the runtime.
LiveUsdCliTextureValidator = _live_usd_cli_texture_validator


def _replace_normalized_image(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def _binding(path: str | Path) -> ExecutionArtifactBinding:
    expanded = Path(path).expanduser()
    if expanded.is_symlink():
        raise ValueError(f"Texture capability input must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise ValueError(f"Texture capability input is not a regular file: {resolved}")
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _load(path: str | Path, model: type[Any]) -> Any:
    return model.model_validate(load_json(Path(path).expanduser().resolve()))


def _print(payload: Any) -> None:
    print(json.dumps(payload.model_dump(mode="json"), sort_keys=True))


def _canonical_json_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _reference(value: str) -> tuple[str, Path]:
    role, separator, raw_path = value.partition("=")
    if not separator or not role.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("reference must use ROLE=PATH")
    return role.strip(), Path(raw_path).expanduser()


def _add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference",
        action="append",
        type=_reference,
        default=[],
        metavar="ROLE=PATH",
        help="Digest-bound reference role and file; may be repeated.",
    )


def add_texture_capability_subcommands(subparsers: Any) -> None:
    """Register focused operations; no command selects or chains another command."""

    agentic_leaf = subparsers.add_parser(
        "agentic-leaf",
        help="Run one focused Texture-owned leaf without compatibility routing.",
    )
    leaf_subparsers = agentic_leaf.add_subparsers(
        dest="texture_agentic_leaf_command",
        required=True,
        metavar="{uv-prepare,prepare,apply-provided,evidence,review,publish}",
    )

    uv_prepare = leaf_subparsers.add_parser(
        "uv-prepare",
        help="Inspect or author exact bounded UVs from one typed invocation.",
    )
    uv_prepare.add_argument("--invocation", required=True, type=Path)
    uv_prepare.set_defaults(handler=_handle_agentic_leaf_uv_prepare)

    focused_handlers = {
        "prepare": _handle_agentic_leaf_prepare,
        "apply-provided": _handle_agentic_leaf_apply_provided,
        "evidence": _handle_agentic_leaf_evidence,
        "review": _handle_agentic_leaf_review,
        "publish": _handle_agentic_leaf_publish,
    }
    for operation, handler in focused_handlers.items():
        parser = leaf_subparsers.add_parser(
            operation,
            help=f"Execute only the typed Texture {operation} graph leaf.",
        )
        parser.add_argument("--invocation", required=True, type=Path)
        parser.set_defaults(handler=handler)

    prepare = subparsers.add_parser(
        "prepare",
        help="Inspect frozen Texture scope and OVRTX evidence without providers.",
    )
    prepare.add_argument("--usd", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--intent", required=True)
    prepare.add_argument("--material-path", action="append", default=[])
    prepare.add_argument("--prim-path", action="append", default=[])
    prepare.add_argument("--texture-size", type=int, default=1024)
    _add_reference_args(prepare)
    for operation in (
        "propose",
        "generate",
        "evidence",
        "critique",
        "review",
        "publish",
    ):
        prepare.add_argument(
            f"--request-{operation}",
            action="store_true",
            help=f"Freeze the optional {operation} capability as requested.",
        )
    prepare.set_defaults(handler=_handle_prepare)

    propose = subparsers.add_parser(
        "propose",
        help="Request an advisory Texture service proposal explicitly.",
    )
    propose.add_argument("--preparation", required=True, type=Path)
    propose.add_argument("--provider-id", required=True)
    propose.add_argument("--texture-agent-url", required=True)
    propose.add_argument("--texture-agent-token-env")
    propose.add_argument("--texture-timeout", type=float, default=1800.0)
    propose.add_argument("--texture-poll-interval", type=float, default=1.0)
    propose.set_defaults(handler=_handle_propose)

    generate = subparsers.add_parser(
        "generate",
        help="Invoke one explicitly selected novel-texture generator leaf.",
    )
    generate.add_argument("--preparation", required=True, type=Path)
    generate.add_argument("--outer-plan", required=True, type=Path)
    generate.add_argument("--provider-proposal", type=Path)
    generate.add_argument("--generator-provider", required=True)
    generate.add_argument("--texture-agent-url", required=True)
    generate.add_argument("--texture-agent-token-env")
    generate.add_argument("--texture-timeout", type=float, default=1800.0)
    generate.add_argument("--texture-poll-interval", type=float, default=1.0)
    generate.set_defaults(handler=_handle_generate)

    apply_provided = subparsers.add_parser(
        "apply-provided",
        help=(
            "Deterministically apply outer-provided generated images without a "
            "Texture provider call."
        ),
    )
    apply_provided.add_argument("--preparation", required=True, type=Path)
    apply_provided.add_argument("--outer-plan", required=True, type=Path)
    apply_provided.add_argument("--provider-proposal", type=Path)
    apply_provided.set_defaults(handler=_handle_apply_provided)

    evidence = subparsers.add_parser(
        "evidence",
        help="Collect static and matched OVRTX candidate evidence without VQA.",
    )
    evidence.add_argument("--preparation", required=True, type=Path)
    evidence.add_argument("--outer-plan", required=True, type=Path)
    evidence.add_argument("--generation", required=True, type=Path)
    evidence.set_defaults(handler=_handle_evidence)

    critique = subparsers.add_parser(
        "critique",
        help="Request advisory domain VLM critique over existing evidence.",
    )
    critique.add_argument("--preparation", required=True, type=Path)
    critique.add_argument("--candidate-evidence", required=True, type=Path)
    critique.add_argument("--vlm-backend", required=True)
    critique.add_argument("--vlm-model", required=True)
    critique.add_argument("--vlm-base-url")
    critique.add_argument("--vlm-api-key-env")
    critique.add_argument("--vlm-timeout", type=float, default=120.0)
    critique.set_defaults(handler=_handle_critique)

    review = subparsers.add_parser(
        "review",
        help="Record an outer-authored review over exact references and renders.",
    )
    review.add_argument("--outer-plan", required=True, type=Path)
    review.add_argument("--generation", required=True, type=Path)
    review.add_argument("--candidate-evidence", required=True, type=Path)
    review.add_argument("--review-input", required=True, type=Path)
    review.add_argument("--critique", type=Path)
    review.set_defaults(handler=_handle_review)

    publish = subparsers.add_parser(
        "publish",
        help="Finalize the exact accepted candidate with deterministic readback.",
    )
    publish.add_argument("--request", required=True, type=Path)
    publish.add_argument("--preparation", required=True, type=Path)
    publish.add_argument("--outer-plan", required=True, type=Path)
    publish.add_argument("--generation", required=True, type=Path)
    publish.add_argument("--candidate-evidence", required=True, type=Path)
    publish.add_argument("--outer-review", required=True, type=Path)
    publish.add_argument("--output-usd", required=True, type=Path)
    publish.set_defaults(handler=_handle_publish)


def _handle_agentic_leaf_uv_prepare(args: argparse.Namespace) -> int:
    from content_agent_workflows.texture.uv_authoring import run_texture_uv_leaf

    result = run_texture_uv_leaf(args.invocation)
    _print(result)
    return 0


def _handle_agentic_leaf_prepare(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    result = texture.run_texture_prepare_asset_leaf(
        args.invocation,
        inspector=LiveUsdCliTextureValidator(
            assessor=None,
            validation_policy_id="not_requested",
        ),
    )
    _print(result)
    return 0


def _handle_agentic_leaf_apply_provided(args: argparse.Namespace) -> int:
    result = _texture_runtime().run_texture_apply_provided_asset_leaf(args.invocation)
    _print(result)
    return 0


def _handle_agentic_leaf_evidence(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    result = texture.run_texture_evidence_asset_leaf(
        args.invocation,
        collector=LiveUsdCliTextureValidator(
            assessor=None,
            validation_policy_id="not_requested",
        ),
    )
    _print(result)
    return 0


def _handle_agentic_leaf_review(args: argparse.Namespace) -> int:
    result = _texture_runtime().run_texture_review_asset_leaf(args.invocation)
    _print(result)
    return 0


def _handle_agentic_leaf_publish(args: argparse.Namespace) -> int:
    result = _texture_runtime().run_texture_publish_asset_leaf(args.invocation)
    _print(result)
    return 0


def _handle_prepare(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    selection = texture.TextureOperationSelection(
        propose="requested" if args.request_propose else "not_requested",
        generate="requested" if args.request_generate else "not_requested",
        evidence="requested" if args.request_evidence else "not_requested",
        critique="requested" if args.request_critique else "not_requested",
        review="requested" if args.request_review else "not_requested",
        publish="requested" if args.request_publish else "not_requested",
    )
    request = texture.build_texture_capability_request(
        source_asset=args.usd,
        output_dir=args.output_dir,
        intent=args.intent,
        operations=selection,
        material_prim_paths=args.material_path,
        prim_paths=args.prim_path,
        reference_artifacts=args.reference,
        texture_size=args.texture_size,
    )
    inspector = LiveUsdCliTextureValidator(
        assessor=None,
        validation_policy_id="not_requested",
    )
    packet, _binding_value = texture.prepare_texture_scope(
        request,
        inspector=inspector,
    )
    _print(packet)
    return 0


def _service_client(args: argparse.Namespace) -> TextureAgentServiceClient:
    from content_workflow_cli.texture_runner import (
        _normalized_url,
        _optional_env_name,
        _validate_bearer_transport,
    )

    base_url = _normalized_url(args.texture_agent_url, "--texture-agent-url")
    token_env = _optional_env_name(
        args.texture_agent_token_env,
        "--texture-agent-token-env",
    )
    _validate_bearer_transport(
        base_url,
        credential_env=token_env,
        option="--texture-agent-url",
    )
    token = None
    if token_env:
        token = os.getenv(token_env)
        if not token:
            raise ValueError(f"{token_env} is not set or is empty")
    return cast(
        "TextureAgentServiceClient",
        _texture_runtime().TextureAgentServiceClient(
            base_url=base_url,
            token=token,
            timeout_seconds=args.texture_timeout,
            poll_interval_seconds=args.texture_poll_interval,
        ),
    )


def _handle_propose(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    if preparation.operation_status.outcome("propose").state == "not_requested":
        raise ValueError("Texture proposal was not requested by the frozen plan")
    packet, _binding_value = texture.request_texture_provider_proposal(
        preparation,
        preparation_binding=_binding(args.preparation),
        provider=_service_client(args),
        provider_id=args.provider_id,
    )
    _print(packet)
    return 0


class _ServiceGeneratorLeaf:
    """Explicit Texture service leaf driven by exact outer-authored semantics."""

    capability_id = "texture-agent-service.execute.v1"

    def __init__(
        self,
        *,
        provider_id: str,
        client: TextureAgentServiceClient,
        endpoint: str | None = None,
        advisory_plan: Any | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.client = client
        if endpoint is not None and (
            not isinstance(endpoint, str) or not endpoint.strip()
        ):
            raise ValueError("Texture generation endpoint must be a non-empty string")
        self.endpoint = endpoint.strip() if endpoint is not None else None
        # Retained only as explicit advisory provenance. Mutation never reads it.
        self.advisory_plan = advisory_plan

    @staticmethod
    def _common_input_value(
        inputs: tuple[TextureGeneratorInputs, ...], field: str
    ) -> Any:
        values = tuple(getattr(item, field) for item in inputs)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"Texture service generator requires one common {field} value"
            )
        return values[0]

    @staticmethod
    def _unit_scope_by_id(
        request: TextureGeneratorLeafRequest,
    ) -> dict[str, dict[str, Any]]:
        scope = {
            unit.unit_id: unit.model_dump(mode="json")
            for unit in request.scope_plan.selected_units
        }
        if tuple(scope) != request.target_unit_ids:
            raise ValueError(
                "deterministic Texture scope differs from outer target order"
            )
        return scope

    def _execution_request(
        self,
        request: TextureGeneratorLeafRequest,
        *,
        unit_scope_by_id: dict[str, dict[str, Any]],
        inputs_by_unit: dict[str, TextureGeneratorInputs],
    ) -> Any:
        texture = _texture_runtime()
        if request.source_dependencies:
            raise ValueError(
                "Texture service generator requires a byte-self-contained source; "
                "the prepared USD dependency closure cannot be uploaded"
            )
        material_textures = (
            texture.TextureAgentServiceClient.material_textures_for_outer_plan(
                request.target_unit_ids,
                unit_scope_by_id=unit_scope_by_id,
                generator_inputs_by_unit=inputs_by_unit,
            )
        )
        scope_payload = request.scope_plan.model_dump(mode="json")
        scope_request = scope_payload.get("request")
        if not isinstance(scope_request, dict):
            raise ValueError("deterministic Texture scope omitted its request")
        metadata: dict[str, Any] = {
            "auto_prompt_enabled": False,
            "material_textures": material_textures,
            "texture_backend": self.provider_id,
            "texture_size": self._common_input_value(
                request.generator_inputs, "texture_size"
            ),
            "detail_policy": self._common_input_value(
                request.generator_inputs, "detail_policy"
            ),
            "discovery_mode": scope_request.get("discovery_mode", "explicit"),
            "unit_mode": scope_request.get("unit_mode", "per_material"),
            "explicit_material_paths": scope_request.get("explicit_material_paths", []),
            "explicit_prim_paths": scope_request.get("explicit_prim_paths", []),
            # Provider-neutral preparation has already established exact UV
            # readiness.  The execution service may validate that same selected
            # scope, but it must not author UVs or widen preparation to the stage.
            "uv_policy": "validate",
            "uv_scope": "target_prims",
            "source_asset_binding": request.source.model_dump(mode="json"),
        }
        engine = self._common_input_value(request.generator_inputs, "engine")
        seed = self._common_input_value(request.generator_inputs, "seed")
        parameters = self._common_input_value(request.generator_inputs, "parameters")
        if engine is not None:
            metadata["backend_engine"] = engine
        if seed is not None:
            metadata["seed"] = seed
        if parameters:
            metadata["backend_custom_parameters"] = parameters
        if self.endpoint is not None:
            metadata["texture_endpoint"] = self.endpoint
        return texture.TextureWorkflowRequest(
            source_asset=request.source.path,
            output_dir=Path(request.output_dir),
            intent=request.intent,
            reference_artifacts=request.reference_artifacts,
            metadata=metadata,
        )

    def _validate_execution_plan(
        self,
        request: TextureGeneratorLeafRequest,
        plan: Any,
        *,
        unit_scope_by_id: dict[str, dict[str, Any]],
    ) -> None:
        if plan.selected_unit_ids != request.target_unit_ids:
            raise ValueError(
                "Texture service execution plan differs from outer target scope"
            )
        for unit in plan.selected_units:
            service_scope = unit.model_dump(mode="json")
            deterministic_scope = unit_scope_by_id[unit.unit_id]
            for field in (
                "material_prim_paths",
                "member_prim_paths",
                "member_subset_paths",
            ):
                if service_scope.get(field, []) != deterministic_scope.get(field, []):
                    raise ValueError(
                        "Texture service execution scope differs from deterministic "
                        f"preparation: {unit.unit_id}.{field}"
                    )
        plan_payload = plan.model_dump(mode="json")
        execution = plan_payload.get("execution")
        if not isinstance(execution, dict):
            raise ValueError("Texture service execution plan omitted provider settings")
        expected_size = self._common_input_value(
            request.generator_inputs, "texture_size"
        )
        if execution.get("backend") != self.provider_id:
            raise ValueError(
                "Texture service execution backend differs from the selected leaf"
            )
        if execution.get("texture_size") != expected_size:
            raise ValueError(
                "Texture service cannot honor the outer texture_size for this session"
            )

    def generate(self, request: TextureGeneratorLeafRequest) -> Any:
        texture = _texture_runtime()
        if any(
            item.execution_mode != "provider_generate" or item.provided_images
            for item in request.generator_inputs
        ):
            raise ValueError("Texture service generator rejects apply_provided inputs")
        if any(item.backend != self.provider_id for item in request.generator_inputs):
            raise ValueError(
                "outer Texture generator backend differs from selected service leaf"
            )
        unit_scope_by_id = self._unit_scope_by_id(request)
        inputs_by_unit = dict(
            zip(request.target_unit_ids, request.generator_inputs, strict=True)
        )
        execution_request = self._execution_request(
            request,
            unit_scope_by_id=unit_scope_by_id,
            inputs_by_unit=inputs_by_unit,
        )
        request_payload = request.model_dump(mode="json")
        request_digest = _canonical_json_digest(request_payload)
        resume_path = Path(request.output_dir) / "texture_service_execution_resume.json"
        if resume_path.exists():
            if resume_path.is_symlink():
                raise ValueError("Texture service resume packet must not be a symlink")
            resume_payload = load_json(resume_path)
            if (
                resume_payload.get("schema_version")
                != "content-workflow-cli.texture-service-execution-resume.v1"
                or resume_payload.get("provider_id") != self.provider_id
                or resume_payload.get("capability_id") != self.capability_id
                or resume_payload.get("request_digest") != request_digest
                or resume_payload.get("reference_artifacts")
                != [
                    item.model_dump(mode="json") for item in request.reference_artifacts
                ]
            ):
                raise ValueError(
                    "Texture service resume packet changed execution identity"
                )
            resume_state_payload = {
                "plan": resume_payload.get("plan"),
                "client_state": resume_payload.get("client_state"),
            }
            if resume_payload.get("resume_state_digest") != _canonical_json_digest(
                resume_state_payload
            ):
                raise ValueError(
                    "Texture service resume state changed after persistence"
                )
            plan = texture.TexturePlanDocument.model_validate(
                resume_state_payload["plan"]
            )
            client_state = resume_payload.get("client_state")
            if not isinstance(client_state, dict):
                raise ValueError("Texture service resume packet omitted client state")
            self.client.restore_resume_state(plan, client_state)
        else:
            plan = self.client.plan(execution_request)

        def persist_resume_state() -> None:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_state_payload = {
                "plan": plan.model_dump(mode="json"),
                "client_state": self.client.export_resume_state(plan),
            }
            atomic_write_json(
                resume_path,
                {
                    "schema_version": (
                        "content-workflow-cli.texture-service-execution-resume.v1"
                    ),
                    "provider_id": self.provider_id,
                    "capability_id": self.capability_id,
                    "request_digest": request_digest,
                    "reference_artifacts": [
                        item.model_dump(mode="json")
                        for item in request.reference_artifacts
                    ],
                    **resume_state_payload,
                    "resume_state_digest": _canonical_json_digest(resume_state_payload),
                },
            )

        if not resume_path.exists():
            persist_resume_state()
        self._validate_execution_plan(
            request,
            plan,
            unit_scope_by_id=unit_scope_by_id,
        )
        result = self.client.execute_outer_plan(
            plan,
            request.target_unit_ids,
            unit_scope_by_id=unit_scope_by_id,
            generator_inputs_by_unit=inputs_by_unit,
            output_dir=Path(request.output_dir),
            preserved_artifacts={},
            persist_resume_state=persist_resume_state,
        )
        persist_resume_state()
        first_artifact, *remaining_artifacts = result.unit_artifacts
        resume_bound_artifact = first_artifact.model_copy(
            update={
                "artifact_paths": (
                    *first_artifact.artifact_paths,
                    str(resume_path.resolve()),
                ),
                "metadata": {
                    **first_artifact.metadata,
                    "texture_service_resume_packet": str(resume_path.resolve()),
                },
            }
        )
        return result.model_copy(
            update={
                "unit_artifacts": (
                    resume_bound_artifact,
                    *remaining_artifacts,
                ),
                "metadata": {
                    **result.metadata,
                    "generator_provider": self.provider_id,
                    "generator_capability": self.capability_id,
                    "outer_generator_inputs": [
                        item.model_dump(mode="json")
                        for item in request.generator_inputs
                    ],
                },
            }
        )


class _RecordedCompanionGeneratorLeaf:
    """Apply exact images already produced by the outer coding-agent companion."""

    provider_id = "coding_agent_companion"
    capability_id = "image-generation+texture.apply-provided.v1"
    _HANDOFF_SCHEMA = "content-workflow-cli.texture-companion-generation-handoff.v1"
    _RESULT_SCHEMA = "agentic-image-generation-result.v1"

    def __init__(
        self,
        handoff: Mapping[str, Any],
        *,
        accepted_plan: ExecutionArtifactBinding,
    ) -> None:
        if handoff.get("schema_version") != self._HANDOFF_SCHEMA:
            raise ValueError("Texture companion generation handoff schema is invalid")
        if handoff.get("backend") != self.provider_id:
            raise ValueError("Texture companion generation handoff backend changed")
        if handoff.get("fallback") is not False:
            raise ValueError("Texture companion generation handoff permits fallback")
        handoff_plan = self._verified_binding(
            handoff.get("accepted_plan"),
            label="Texture companion accepted plan",
        )
        if handoff_plan != accepted_plan:
            raise ValueError(
                "Texture companion handoff differs from the accepted outer plan"
            )
        raw_units = handoff.get("units")
        if not isinstance(raw_units, list) or not raw_units:
            raise ValueError("Texture companion generation handoff has no units")
        unit_records: dict[str, Mapping[str, Any]] = {}
        for raw in raw_units:
            if not isinstance(raw, Mapping):
                raise ValueError("Texture companion generation unit must be an object")
            unit_id = raw.get("unit_id")
            if not isinstance(unit_id, str) or unit_id in unit_records:
                raise ValueError("Texture companion generation unit IDs are invalid")
            unit_records[unit_id] = raw
        self._handoff = dict(handoff)
        self._unit_records = unit_records

    @staticmethod
    def _verified_binding(raw: Any, *, label: str) -> ExecutionArtifactBinding:
        binding = ExecutionArtifactBinding.model_validate(raw)
        observed = _binding(binding.path)
        if observed != binding:
            raise ValueError(f"{label} bytes changed")
        return binding

    @staticmethod
    def _validate_conditioning_images(
        raw: Any,
        expected: tuple[ExecutionArtifactBinding, ...],
    ) -> None:
        if not isinstance(raw, list) or len(raw) != len(expected):
            raise ValueError("Texture companion conditioning-image count changed")
        for record, binding in zip(raw, expected, strict=True):
            if not isinstance(record, Mapping):
                raise ValueError("Texture companion conditioning record is invalid")
            if (
                record.get("path") != binding.path
                or record.get("sha256") != binding.sha256
            ):
                raise ValueError("Texture companion conditioning image changed")
            if _binding(binding.path) != binding:
                raise ValueError("Texture companion conditioning-image bytes changed")

    def _provided_image(
        self,
        *,
        unit_id: str,
        inputs: TextureGeneratorInputs,
    ) -> tuple[
        Any,
        ExecutionArtifactBinding,
        ExecutionArtifactBinding,
        ExecutionArtifactBinding | None,
    ]:
        texture = _texture_runtime()
        record = self._unit_records[unit_id]
        prompt = self._verified_binding(
            record.get("prompt"),
            label="Texture companion prompt",
        )
        if Path(prompt.path).read_text(encoding="utf-8") != inputs.prompt:
            raise ValueError("Texture companion prompt differs from accepted inputs")
        expected_references = tuple(inputs.reference_artifacts)
        if record.get("conditioning_images") != [
            item.model_dump(mode="json") for item in expected_references
        ]:
            raise ValueError("Texture companion handoff references changed")
        manifest_path = Path(str(record.get("manifest_path") or "")).expanduser()
        manifest_binding = _binding(manifest_path)
        manifest = load_json(Path(manifest_binding.path))
        if (
            manifest.get("schema_version") != self._RESULT_SCHEMA
            or manifest.get("status") != "completed"
            or manifest.get("mode") != "coding_agent_companion"
        ):
            raise ValueError("Texture companion image generation did not complete")
        request = manifest.get("request")
        if not isinstance(request, Mapping):
            raise ValueError("Texture companion image manifest omitted its request")
        if (
            request.get("prompt_file") != prompt.path
            or request.get("prompt_sha256") != prompt.sha256
        ):
            raise ValueError("Texture companion image manifest changed its prompt")
        self._validate_conditioning_images(
            request.get("conditioning_images"),
            expected_references,
        )
        provider = manifest.get("provider")
        if not isinstance(provider, Mapping):
            raise ValueError("Texture companion image manifest omitted its provider")
        tool_id = provider.get("tool_id")
        record_tool_id = record.get("tool_id")
        if (
            not isinstance(tool_id, str)
            or not tool_id.strip()
            or tool_id != tool_id.strip()
            or not isinstance(record_tool_id, str)
            or not record_tool_id.strip()
            or record_tool_id != record_tool_id.strip()
            or tool_id != record_tool_id
            or provider.get("backend") is not None
        ):
            raise ValueError("Texture companion image provider identity changed")
        output = manifest.get("output")
        if not isinstance(output, Mapping):
            raise ValueError("Texture companion image manifest omitted its output")
        output_binding = _binding(str(record.get("output_path") or ""))
        texture_size = int(record.get("texture_size") or 0)
        if (
            output.get("path") != output_binding.path
            or output.get("sha256") != output_binding.sha256
            or output.get("media_type") != "image/png"
            or texture_size != inputs.texture_size
        ):
            raise ValueError("Texture companion image output differs from the handoff")
        try:
            with Image.open(output_binding.path) as image:
                image.load()
                observed_size = image.size
                observed_format = image.format
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("Texture companion output is not a safe image") from exc
        if (
            observed_format != "PNG"
            or output.get("width") != observed_size[0]
            or output.get("height") != observed_size[1]
            or observed_size[0] != observed_size[1]
        ):
            raise ValueError("Texture companion raw image dimensions are invalid")
        normalization_binding: ExecutionArtifactBinding | None = None
        applied_binding = output_binding
        if observed_size != (texture_size, texture_size):
            normalized_path = manifest_path.parent / f"normalized-{texture_size}.png"
            normalization_path = manifest_path.parent / "texture_normalization.json"
            expected_source = output_binding.model_dump(mode="json")
            if normalization_path.exists():
                normalization_binding = _binding(normalization_path)
                normalization = load_json(Path(normalization_binding.path))
                normalized_binding = self._verified_binding(
                    normalization.get("output"),
                    label="Texture normalized companion image",
                )
                if (
                    normalization.get("schema_version")
                    != "content-workflow-cli.texture-image-normalization.v1"
                    or normalization.get("source") != expected_source
                    or normalization.get("target_size") != texture_size
                    or normalization.get("method") != "lanczos-rgb-png"
                    or normalized_binding.path != str(normalized_path.resolve())
                ):
                    raise ValueError("Texture companion normalization identity changed")
            else:
                if normalized_path.exists():
                    raise FileExistsError(
                        "Texture normalized image exists without its receipt"
                    )
                temporary = normalized_path.with_suffix(".tmp.png")
                if temporary.exists():
                    raise FileExistsError(
                        "Texture companion normalization temporary file exists"
                    )
                try:
                    with Image.open(output_binding.path) as image:
                        image.convert("RGB").resize(
                            (texture_size, texture_size),
                            Image.Resampling.LANCZOS,
                        ).save(temporary, format="PNG")
                    _replace_normalized_image(temporary, normalized_path)
                finally:
                    temporary.unlink(missing_ok=True)
                normalized_binding = _binding(normalized_path)
                atomic_write_json(
                    normalization_path,
                    {
                        "schema_version": (
                            "content-workflow-cli.texture-image-normalization.v1"
                        ),
                        "source": expected_source,
                        "output": normalized_binding.model_dump(mode="json"),
                        "source_size": list(observed_size),
                        "target_size": texture_size,
                        "method": "lanczos-rgb-png",
                        "semantic_edit": False,
                    },
                )
                normalization_binding = _binding(normalization_path)
            applied_binding = normalized_binding
        producer = texture.TextureProvidedImageProducer(
            provider=tool_id,
            capability="image-generation.record-companion.v1",
            invocation_id=f"companion-{manifest_binding.sha256[:20]}",
            provenance={
                "mode": "coding_agent_companion",
                "model": provider.get("model"),
                "image_generation_manifest": manifest_binding.model_dump(mode="json"),
            },
        )
        provided = texture.TextureProvidedImageArtifact(
            unit_id=unit_id,
            channel="albedo",
            artifact=applied_binding,
            producer=producer,
        )
        return provided, prompt, manifest_binding, normalization_binding

    def generate(self, request: TextureGeneratorLeafRequest) -> Any:
        texture = _texture_runtime()
        if tuple(self._unit_records) != request.target_unit_ids:
            raise ValueError("Texture companion handoff unit order changed")
        transformed_inputs: list[TextureGeneratorInputs] = []
        companion_artifacts: dict[
            str,
            tuple[
                ExecutionArtifactBinding,
                ExecutionArtifactBinding,
                ExecutionArtifactBinding | None,
            ],
        ] = {}
        for unit_id, inputs in zip(
            request.target_unit_ids,
            request.generator_inputs,
            strict=True,
        ):
            if (
                inputs.execution_mode != "provider_generate"
                or inputs.backend != self.provider_id
                or inputs.provided_images
            ):
                raise ValueError(
                    "Texture companion generator received non-companion inputs"
                )
            provided, prompt, manifest, normalization = self._provided_image(
                unit_id=unit_id,
                inputs=inputs,
            )
            transformed_inputs.append(
                inputs.model_copy(
                    update={
                        "execution_mode": "apply_provided",
                        "backend": texture.ProvidedImageTextureApplyLeaf.provider_id,
                        "engine": None,
                        "seed": None,
                        "parameters": {},
                        "provided_images": (provided,),
                    }
                )
            )
            companion_artifacts[unit_id] = (prompt, manifest, normalization)
        applied = texture.ProvidedImageTextureApplyLeaf().generate(
            request.model_copy(update={"generator_inputs": tuple(transformed_inputs)})
        )
        unit_artifacts = tuple(
            item.model_copy(
                update={
                    "artifact_paths": (
                        *item.artifact_paths,
                        companion_artifacts[item.unit_id][0].path,
                        companion_artifacts[item.unit_id][1].path,
                        *(
                            (companion_artifacts[item.unit_id][2].path,)
                            if companion_artifacts[item.unit_id][2] is not None
                            else ()
                        ),
                    ),
                    "metadata": {
                        **item.metadata,
                        "backend": self.provider_id,
                        "capability": self.capability_id,
                        "execution_mode": "provider_generate",
                        "companion_generation_recorded": True,
                        "image_generation_manifest": companion_artifacts[item.unit_id][
                            1
                        ].model_dump(mode="json"),
                        "normalization_receipt": (
                            companion_artifacts[item.unit_id][2].model_dump(mode="json")
                            if companion_artifacts[item.unit_id][2] is not None
                            else None
                        ),
                        "application_capability": (
                            texture.ProvidedImageTextureApplyLeaf.capability_id
                        ),
                    },
                }
            )
            for item in applied.unit_artifacts
        )
        return applied.model_copy(
            update={
                "unit_artifacts": unit_artifacts,
                "metadata": {
                    **applied.metadata,
                    "backend": self.provider_id,
                    "capability": self.capability_id,
                    "execution_mode": "provider_generate",
                    "provider_invoked": True,
                    "model_invoked": True,
                    "texture_service_constructed": False,
                    "companion_generation_recorded": True,
                },
            }
        )


def _handle_generate(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    outer_plan = _load(args.outer_plan, texture.TextureOuterPlan)
    proposal = (
        _load(args.provider_proposal, texture.TextureProviderProposalPacket)
        if args.provider_proposal
        else None
    )
    if preparation.operation_status.outcome("generate").state == "not_requested":
        raise ValueError("Texture generation was not requested by the frozen plan")
    generator = _ServiceGeneratorLeaf(
        provider_id=args.generator_provider,
        client=_service_client(args),
        endpoint=preparation.request.metadata.get("texture_endpoint"),
        advisory_plan=(proposal.proposal if proposal is not None else None),
    )
    packet, _binding_value = texture.invoke_texture_generator(
        outer_plan,
        outer_plan_binding=_binding(args.outer_plan),
        preparation=preparation,
        preparation_binding=_binding(args.preparation),
        generator=generator,
        proposal=proposal,
        provider_proposal=(
            _binding(args.provider_proposal) if args.provider_proposal else None
        ),
    )
    _print(packet)
    return 0


def _handle_apply_provided(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    outer_plan = _load(args.outer_plan, texture.TextureOuterPlan)
    proposal = (
        _load(args.provider_proposal, texture.TextureProviderProposalPacket)
        if args.provider_proposal
        else None
    )
    if preparation.operation_status.outcome("generate").state == "not_requested":
        raise ValueError("Texture mutation was not requested by the frozen plan")
    if outer_plan.execution_mode != "apply_provided":
        raise ValueError("texture apply-provided requires apply_provided outer mode")
    packet, _binding_value = texture.invoke_texture_generator(
        outer_plan,
        outer_plan_binding=_binding(args.outer_plan),
        preparation=preparation,
        preparation_binding=_binding(args.preparation),
        generator=texture.ProvidedImageTextureApplyLeaf(),
        proposal=proposal,
        provider_proposal=(
            _binding(args.provider_proposal) if args.provider_proposal else None
        ),
    )
    _print(packet)
    return 0


def _handle_evidence(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    outer_plan = _load(args.outer_plan, texture.TextureOuterPlan)
    generation = _load(args.generation, texture.TextureGenerationPacket)
    if generation.operation_status.outcome("evidence").state == "not_requested":
        raise ValueError("Texture evidence was not requested by the frozen plan")
    collector = LiveUsdCliTextureValidator(
        assessor=None,
        validation_policy_id="not_requested",
    )
    packet, _binding_value = texture.collect_texture_candidate_evidence(
        outer_plan,
        generation,
        preparation=preparation,
        preparation_binding=_binding(args.preparation),
        outer_plan_binding=_binding(args.outer_plan),
        generation_binding=_binding(args.generation),
        collector=collector,
    )
    _print(packet)
    return 0


def _handle_critique(args: argparse.Namespace) -> int:
    from content_workflow_cli.texture_runner import (
        PUBLIC_VLM_BACKENDS,
        _create_vlm,
        _first_present_environment_value,
        _normalized_url,
        _optional_env_name,
        _resolve_vlm_api_key,
    )

    texture = _texture_runtime()
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    evidence = _load(args.candidate_evidence, texture.TextureCandidateEvidencePacket)
    if evidence.operation_status.outcome("critique").state == "not_requested":
        raise ValueError("Texture critique was not requested by the frozen plan")
    preparation_binding = _binding(args.preparation)
    if evidence.preparation != preparation_binding:
        raise ValueError("Texture critique preparation differs from candidate evidence")
    backend = str(args.vlm_backend).strip()
    if backend not in PUBLIC_VLM_BACKENDS:
        choices = ", ".join(PUBLIC_VLM_BACKENDS)
        raise ValueError(f"--vlm-backend must be one of: {choices}.")
    model = str(args.vlm_model).strip()
    if not model:
        raise ValueError("--vlm-model must not be empty")
    config: dict[str, Any] = {
        "model": model,
        "timeout": args.vlm_timeout,
    }
    raw_base_url = args.vlm_base_url
    base_url_source = "--vlm-base-url"
    if raw_base_url is None and backend == "openai":
        environment_names = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
        raw_base_url = _first_present_environment_value(*environment_names)
        if raw_base_url is not None:
            base_url_source = next(
                name for name in environment_names if os.environ.get(name, "").strip()
            )
    if raw_base_url is None and backend == "anthropic":
        environment_names = ("ANTHROPIC_API_URL", "ANTHROPIC_BASE_URL")
        raw_base_url = _first_present_environment_value(*environment_names)
        if raw_base_url is not None:
            base_url_source = next(
                name for name in environment_names if os.environ.get(name, "").strip()
            )
    if raw_base_url:
        config["base_url"] = _normalized_url(raw_base_url, "--vlm-base-url")
    api_key_env = _optional_env_name(args.vlm_api_key_env, "--vlm-api-key-env")
    if raw_base_url and backend in {"anthropic", "gemini"} and not api_key_env:
        raise ValueError(
            f"--vlm-api-key-env is required with a custom base URL from "
            f"{base_url_source} for "
            f"{backend}; provider-default credentials are not forwarded to "
            "custom endpoints."
        )
    credential_config = {
        **config,
        "api_key_env": api_key_env,
    }
    api_key = _resolve_vlm_api_key(
        backend,
        credential_config,
        "Texture critique VLM",
    )
    if api_key:
        config["api_key"] = api_key
    assessor = texture.VlmTextureVisualAssessor(_create_vlm(backend=backend, **config))
    contexts = {
        unit.unit_id: unit.model_dump(mode="json")
        for unit in preparation.scope_plan.selected_units
    }
    provider = texture.AssessorTextureCritiqueProvider(
        assessor=assessor,
        provider_id=backend,
        capability_id=f"{backend}:{model}",
        unit_context_by_id=contexts,
    )
    packet, _binding_value = texture.request_texture_critique(
        evidence,
        evidence_binding=_binding(args.candidate_evidence),
        intent=preparation.request.intent,
        provider=provider,
    )
    _print(packet)
    return 0


def _handle_review(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    outer_plan = _load(args.outer_plan, texture.TextureOuterPlan)
    generation = _load(args.generation, texture.TextureGenerationPacket)
    evidence = _load(args.candidate_evidence, texture.TextureCandidateEvidencePacket)
    review_input = _load(args.review_input, texture.TextureOuterReviewInput)
    critique = (
        _load(args.critique, texture.TextureCritiquePacket) if args.critique else None
    )
    packet, _binding_value = texture.record_texture_outer_review(
        review_input,
        outer_plan=outer_plan,
        generation=generation,
        evidence=evidence,
        critique=critique,
        critique_binding=(_binding(args.critique) if args.critique else None),
    )
    _print(packet)
    return 0


def _handle_publish(args: argparse.Namespace) -> int:
    texture = _texture_runtime()
    _request = _load(args.request, texture.TextureCapabilityRequest)
    preparation = _load(args.preparation, texture.TexturePreparationPacket)
    outer_plan = _load(args.outer_plan, texture.TextureOuterPlan)
    generation = _load(args.generation, texture.TextureGenerationPacket)
    evidence = _load(args.candidate_evidence, texture.TextureCandidateEvidencePacket)
    review = _load(args.outer_review, texture.TextureOuterReviewPacket)
    packet, _binding_value = texture.publish_texture_candidate(
        review,
        review_binding=_binding(args.outer_review),
        request_binding=_binding(args.request),
        preparation=preparation,
        preparation_binding=_binding(args.preparation),
        outer_plan=outer_plan,
        outer_plan_binding=_binding(args.outer_plan),
        generation=generation,
        generation_binding=_binding(args.generation),
        evidence=evidence,
        evidence_binding=_binding(args.candidate_evidence),
        publication_path=args.output_usd,
    )
    _print(packet)
    return 0


__all__ = ["add_texture_capability_subcommands"]
