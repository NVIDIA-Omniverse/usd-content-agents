# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI surface for independently selected Articulation capability leaves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from content_agent_workflows.common.artifacts import _stable_ctime_ns

if TYPE_CHECKING:
    from content_agent_workflows.common.domain_execution import (
        ExecutionArtifactBinding,
    )


def add_articulation_capability_subcommands(subparsers: Any) -> None:
    """Register public leaves without selecting or chaining a workflow."""

    agentic_leaf = subparsers.add_parser(
        "agentic-leaf",
        help="Run one focused Articulation leaf without a nested coordinator.",
    )
    leaf_subparsers = agentic_leaf.add_subparsers(
        dest="articulation_agentic_leaf_command",
        required=True,
        metavar="{author,evidence,review,publish}",
    )
    for operation, handler in (
        ("author", _handle_articulation_agentic_leaf_author),
        ("evidence", _handle_articulation_agentic_leaf_evidence),
        ("review", _handle_articulation_agentic_leaf_review),
        ("publish", _handle_articulation_agentic_leaf_publish),
    ):
        parser = leaf_subparsers.add_parser(
            operation,
            help=f"Execute only the typed Articulation {operation} graph leaf.",
        )
        parser.add_argument("--invocation", required=True, type=Path)
        parser.set_defaults(handler=handler)

    publish_preparation = subparsers.add_parser(
        "publish-preparation",
        help="Publish deterministic provider-neutral Articulation preparation.",
    )
    publish_preparation.add_argument("--invocation")
    publish_preparation.add_argument("--readback")
    publish_preparation.add_argument("--retained-root")
    publish_preparation.add_argument("--output-dir")
    publish_preparation.set_defaults(handler=_handle_publish_preparation)

    validate_preparation = subparsers.add_parser(
        "validate-preparation",
        help="Revalidate a sealed Articulation preparation publication.",
    )
    validate_preparation.add_argument("--publication", required=True)
    validate_preparation.set_defaults(handler=_handle_validate_preparation)

    validate_attempt = subparsers.add_parser(
        "validate-attempt",
        help="Revalidate a terminal Articulation provider attempt and lineage.",
    )
    validate_attempt.add_argument("--terminal-receipt", required=True)
    validate_attempt.set_defaults(handler=_handle_validate_attempt)

    propose = subparsers.add_parser(
        "propose",
        help="Invoke one explicit advisory Articulation proposal provider.",
    )
    propose.add_argument("--invocation")
    propose.add_argument("--preparation")
    propose.add_argument("--output-dir")
    propose.add_argument("--intent")
    propose.add_argument(
        "--provider-adapter",
        choices=("artifact-json", "http-json"),
    )
    propose.add_argument("--provider-id")
    propose.add_argument("--capability-id")
    propose.add_argument("--provider-payload")
    propose.add_argument("--provider-url")
    propose.add_argument("--provider-endpoint-alias")
    propose.add_argument("--provider-token-env")
    propose.add_argument("--provider-timeout", type=float)
    propose.add_argument("--replaces-terminal-receipt")
    propose.add_argument("--replacement-reason")
    propose.set_defaults(handler=_handle_articulation_propose)

    bind_output = subparsers.add_parser(
        "bind-output-evidence",
        help="Bind canonical post-authoring OVRTX evidence for exact outer review.",
    )
    bind_output.add_argument("--run-dir", required=True)
    bind_output.add_argument("--canonical-visual-envelope", required=True)
    bind_output.set_defaults(handler=_handle_articulation_bind_output_evidence)

    graph_apply = subparsers.add_parser(
        "project-graph-apply",
        help="Project exact completed graph apply/readback into shared Validation.",
    )
    graph_apply.add_argument("--run-dir", required=True)
    graph_apply.add_argument("--output-dir", required=True)
    graph_apply.set_defaults(handler=_handle_articulation_project_graph_apply)

    for name, handler, help_text in (
        (
            "project-gate3a",
            _handle_articulation_project_gate3a,
            "Verify and project one retained Joint Gate 3A result.",
        ),
        (
            "project-gate3b",
            _handle_articulation_project_gate3b,
            "Verify and project one retained Joint Gate 3B result.",
        ),
    ):
        gate = subparsers.add_parser(name, help=help_text)
        gate.add_argument("--source", required=True)
        gate.add_argument("--output", required=True)
        gate.add_argument("--report", required=True)
        gate.add_argument("--closeout", required=True)
        gate.add_argument("--run-plan", required=True)
        gate.add_argument("--intake", required=True)
        gate.add_argument("--authoring-receipt", required=True)
        gate.add_argument("--output-dir", required=True)
        gate.set_defaults(handler=handler)

    dynamic = subparsers.add_parser(
        "project-dynamic",
        help="Re-verify captured Joint dynamic evidence and project its result.",
    )
    dynamic.add_argument("--source", required=True)
    dynamic.add_argument("--output", required=True)
    dynamic.add_argument("--receipt", required=True)
    dynamic.add_argument("--profile-id", required=True)
    dynamic.add_argument("--artifact-map", required=True)
    dynamic.add_argument("--output-dir", required=True)
    dynamic.set_defaults(handler=_handle_articulation_project_dynamic)


def _open_selected_leaf_invocation(candidate: Path) -> int:
    """Open one absolute file through a held symlink-free directory chain."""

    from world_understanding.utils.artifacts import (
        ArtifactPathError,
        open_regular_file_no_follow,
    )

    try:
        with open_regular_file_no_follow(candidate) as (source, _metadata):
            return os.dup(source.fileno())
    except ArtifactPathError as exc:
        if str(exc) == "Artifact source must be a regular file":
            raise ValueError(
                "selected-leaf invocation must be a bounded unlinked file"
            ) from exc
        raise OSError(f"Could not open selected-leaf invocation: {candidate}") from exc


def _selected_leaf_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Normalize Windows ctime before same-handle identity comparisons."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        _stable_ctime_ns(metadata),
    )


def _load_selected_leaf_invocation_with_binding(
    path: str,
    model: Any,
) -> tuple[Any, ExecutionArtifactBinding]:
    """Parse and bind exact bounded bytes from one held safe descriptor."""

    candidate = Path(os.path.abspath(Path(path).expanduser()))
    descriptor = -1
    verification_descriptor = -1
    try:
        descriptor = _open_selected_leaf_invocation(candidate)
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != 1
            or initial.st_size > 4 * 1024 * 1024
        ):
            raise ValueError("selected-leaf invocation must be a bounded unlinked file")
        chunks: list[bytes] = []
        captured_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            captured_size += len(chunk)
            if captured_size > 4 * 1024 * 1024:
                raise ValueError("selected-leaf invocation exceeds its size bound")
            chunks.append(chunk)
        final = os.fstat(descriptor)
        verification_descriptor = _open_selected_leaf_invocation(candidate)
        path_metadata = os.fstat(verification_descriptor)
    except OSError as exc:
        raise ValueError(
            f"selected-leaf invocation is unavailable: {candidate}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if verification_descriptor >= 0:
            os.close(verification_descriptor)
    if (
        _selected_leaf_identity(initial) != _selected_leaf_identity(final)
        or captured_size != initial.st_size
        or path_metadata.st_dev != initial.st_dev
        or path_metadata.st_ino != initial.st_ino
        or not stat.S_ISREG(path_metadata.st_mode)
        or path_metadata.st_nlink != 1
    ):
        raise ValueError("selected-leaf invocation changed during held-descriptor read")
    document = b"".join(chunks)
    try:
        invocation = model.model_validate_json(document)
    except ValueError as exc:
        raise ValueError(f"selected-leaf invocation is invalid: {exc}") from exc
    from content_agent_workflows.common.domain_execution import (
        ExecutionArtifactBinding,
    )

    binding = ExecutionArtifactBinding(
        path=str(candidate),
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
    )
    return invocation, binding


def _load_selected_leaf_invocation(path: str, model: Any) -> Any:
    """Parse exact bounded bytes from one held symlink-safe descriptor."""

    invocation, _binding = _load_selected_leaf_invocation_with_binding(path, model)
    return invocation


def _verify_selected_leaf_binding(binding: Any, *, label: str) -> None:
    """Re-establish one invocation-bound artifact before domain execution."""

    from content_agent_workflows.common.artifacts import file_sha256

    candidate = Path(os.path.abspath(Path(binding.path).expanduser()))
    try:
        metadata = candidate.lstat()
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable: {candidate}") from exc
    if (
        resolved != candidate
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != binding.size_bytes
        or file_sha256(resolved) != binding.sha256
    ):
        raise ValueError(f"{label} differs from the selected-leaf invocation")


def _selected_leaf_output_dir(invocation_path: str, leaf_name: str) -> str:
    """Confine selected output beside the host-chosen invocation envelope."""

    invocation = Path(os.path.abspath(Path(invocation_path).expanduser()))
    return str(invocation.parent / leaf_name)


def _run_articulation_agentic_leaf(
    args: argparse.Namespace,
    *,
    runner_name: str,
) -> int:
    """Load and execute only the selected deterministic runtime leaf."""

    import content_agent_workflows.articulation as articulation

    runner = getattr(articulation, runner_name)
    result = runner(args.invocation)
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_articulation_agentic_leaf_author(args: argparse.Namespace) -> int:
    return _run_articulation_agentic_leaf(
        args,
        runner_name="run_articulation_author_asset_leaf",
    )


def _handle_articulation_agentic_leaf_evidence(args: argparse.Namespace) -> int:
    return _run_articulation_agentic_leaf(
        args,
        runner_name="run_articulation_evidence_asset_leaf",
    )


def _handle_articulation_agentic_leaf_review(args: argparse.Namespace) -> int:
    return _run_articulation_agentic_leaf(
        args,
        runner_name="run_articulation_review_asset_leaf",
    )


def _handle_articulation_agentic_leaf_publish(args: argparse.Namespace) -> int:
    return _run_articulation_agentic_leaf(
        args,
        runner_name="run_articulation_publish_asset_leaf",
    )


def _handle_articulation_propose(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import (
        ArticulationProposalAttemptFailed,
        ArticulationProposalLeafInvocation,
        ArticulationProposalProvider,
        ArtifactJsonArticulationProposalProvider,
        HttpJsonArticulationProposalProvider,
        request_embedded_articulation_provider_proposal,
        validate_articulation_proposal_attempt_receipt,
    )

    if args.invocation is not None:
        if any(
            value is not None
            for value in (
                args.preparation,
                args.output_dir,
                args.intent,
                args.provider_adapter,
                args.provider_id,
                args.capability_id,
                args.provider_payload,
                args.provider_url,
                args.provider_endpoint_alias,
                args.provider_token_env,
                args.provider_timeout,
                args.replaces_terminal_receipt,
                args.replacement_reason,
            )
        ):
            raise ValueError(
                "--invocation cannot be mixed with direct proposal options"
            )
        invocation = _load_selected_leaf_invocation(
            args.invocation,
            ArticulationProposalLeafInvocation,
        )
        _verify_selected_leaf_binding(
            invocation.preparation,
            label="selected Articulation preparation",
        )
        preparation = invocation.preparation.path
        output_dir = _selected_leaf_output_dir(
            args.invocation,
            "articulation-proposal-attempt",
        )
        intent = invocation.intent
        provider_adapter = invocation.provider.adapter
        provider_id = invocation.provider.provider_id
        capability_id = invocation.provider.capability_id
        provider_payload = invocation.provider.provider_payload.path
        _verify_selected_leaf_binding(
            invocation.provider.provider_payload,
            label="selected proposal provider payload",
        )
        provider_url = None
        provider_endpoint_alias = None
        provider_token_env = None
        provider_timeout = None
        replaces_terminal_receipt = (
            invocation.replaces_terminal_receipt.path
            if invocation.replaces_terminal_receipt is not None
            else None
        )
        expected_preparation = invocation.preparation
        expected_provider_payload = invocation.provider.provider_payload
        expected_replacement_terminal = invocation.replaces_terminal_receipt
        if expected_replacement_terminal is not None:
            _verify_selected_leaf_binding(
                expected_replacement_terminal,
                label="selected replacement terminal receipt",
            )
        replacement_reason = invocation.replacement_reason
    else:
        required = {
            "--preparation": args.preparation,
            "--output-dir": args.output_dir,
            "--intent": args.intent,
            "--provider-adapter": args.provider_adapter,
            "--provider-id": args.provider_id,
            "--capability-id": args.capability_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"direct proposal invocation requires {', '.join(missing)}"
            )
        preparation = args.preparation
        output_dir = args.output_dir
        intent = args.intent
        provider_adapter = args.provider_adapter
        provider_id = args.provider_id
        capability_id = args.capability_id
        provider_payload = args.provider_payload
        provider_url = args.provider_url
        provider_endpoint_alias = args.provider_endpoint_alias
        provider_token_env = args.provider_token_env
        provider_timeout = args.provider_timeout
        replaces_terminal_receipt = args.replaces_terminal_receipt
        replacement_reason = args.replacement_reason
        expected_preparation = None
        expected_provider_payload = None
        expected_replacement_terminal = None

    provider: ArticulationProposalProvider
    if provider_adapter == "artifact-json":
        if not provider_payload:
            raise ValueError("artifact-json requires --provider-payload")
        if any(
            value is not None
            for value in (
                provider_url,
                provider_endpoint_alias,
                provider_token_env,
                provider_timeout,
            )
        ):
            raise ValueError("artifact-json cannot be mixed with HTTP provider options")
        provider = ArtifactJsonArticulationProposalProvider(
            provider_id=provider_id,
            capability_id=capability_id,
            payload_path=provider_payload,
            expected_payload=expected_provider_payload,
        )
    else:
        if provider_payload is not None:
            raise ValueError("http-json cannot be mixed with --provider-payload")
        if not provider_url or not provider_endpoint_alias:
            raise ValueError(
                "http-json requires --provider-url and --provider-endpoint-alias"
            )
        token = None
        if provider_token_env:
            token = os.getenv(provider_token_env)
            if not token:
                raise ValueError(f"{provider_token_env} is not set or is empty")
        provider = HttpJsonArticulationProposalProvider(
            provider_id=provider_id,
            capability_id=capability_id,
            endpoint_alias=provider_endpoint_alias,
            endpoint_url=provider_url,
            bearer_token=token,
            timeout_seconds=(
                provider_timeout if provider_timeout is not None else 120.0
            ),
        )
    try:
        publication = request_embedded_articulation_provider_proposal(
            preparation,
            output_dir=output_dir,
            intent=intent,
            provider=provider,
            replaces_terminal_receipt=replaces_terminal_receipt,
            replacement_reason=replacement_reason,
            expected_preparation=expected_preparation,
            expected_replacement_terminal=expected_replacement_terminal,
        )
    except ArticulationProposalAttemptFailed as exc:
        if args.invocation is not None:
            terminal = validate_articulation_proposal_attempt_receipt(
                exc.receipt_binding.path
            )
            print(json.dumps(terminal.model_dump(mode="json"), sort_keys=True))
            return 1
        print(
            json.dumps(
                {
                    "status": "failed",
                    "terminal_receipt": exc.receipt_binding.model_dump(mode="json"),
                    "disposition": exc.receipt.disposition,
                    "failure_code": exc.receipt.failure_code,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    result = (
        validate_articulation_proposal_attempt_receipt(
            publication.terminal_receipt.path
        )
        if args.invocation is not None
        else publication
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_publish_preparation(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import (
        ArticulationPreparationLeafInvocation,
        EmbeddedArticulationError,
        fail_articulation_preparation_asset_leaf,
        publish_embedded_articulation_preparation,
    )

    if args.invocation is not None:
        if any(
            value is not None
            for value in (args.readback, args.retained_root, args.output_dir)
        ):
            raise ValueError(
                "--invocation cannot be mixed with direct preparation options"
            )
        invocation, invocation_binding = _load_selected_leaf_invocation_with_binding(
            args.invocation,
            ArticulationPreparationLeafInvocation,
        )
        failure_receipt = (
            Path(invocation_binding.path).parent
            / "articulation_preparation_failure_receipt.json"
        )
        if failure_receipt.exists() or failure_receipt.is_symlink():
            terminal = fail_articulation_preparation_asset_leaf(
                args.invocation,
                expected_invocation_binding=invocation_binding,
            )
            print(json.dumps(terminal.model_dump(mode="json"), sort_keys=True))
            return 1
        readback = invocation.readback.path
        retained_root = invocation.retained_root
        output_dir = _selected_leaf_output_dir(
            args.invocation,
            "articulation-preparation-publication",
        )
        expected_readback = invocation.readback
        try:
            _verify_selected_leaf_binding(
                invocation.readback,
                label="selected preparation readback",
            )
            publication = publish_embedded_articulation_preparation(
                readback,
                retained_root=retained_root,
                output_dir=output_dir,
                expected_readback=expected_readback,
            )
        except (EmbeddedArticulationError, ValueError):
            publication_root = Path(output_dir)
            if publication_root.exists() or publication_root.is_symlink():
                raise
            terminal = fail_articulation_preparation_asset_leaf(
                args.invocation,
                expected_invocation_binding=invocation_binding,
            )
            print(json.dumps(terminal.model_dump(mode="json"), sort_keys=True))
            return 1
    else:
        required = {
            "--readback": args.readback,
            "--retained-root": args.retained_root,
            "--output-dir": args.output_dir,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"direct preparation invocation requires {', '.join(missing)}"
            )
        readback = args.readback
        retained_root = args.retained_root
        output_dir = args.output_dir
        expected_readback = None
        publication = publish_embedded_articulation_preparation(
            readback,
            retained_root=retained_root,
            output_dir=output_dir,
            expected_readback=expected_readback,
        )
    print(json.dumps(publication.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_validate_preparation(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import (
        validate_embedded_articulation_preparation_publication,
    )

    publication = validate_embedded_articulation_preparation_publication(
        args.publication
    )
    print(json.dumps(publication.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_validate_attempt(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import (
        validate_articulation_proposal_attempt_receipt,
    )

    receipt = validate_articulation_proposal_attempt_receipt(args.terminal_receipt)
    print(json.dumps(receipt.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_articulation_bind_output_evidence(args: argparse.Namespace) -> int:
    from .articulation_runner import bind_articulation_output_evidence_step

    result = bind_articulation_output_evidence_step(
        args.run_dir,
        args.canonical_visual_envelope,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


def _print_projector_publication(publication: Any) -> int:
    print(json.dumps(publication.model_dump(mode="json"), sort_keys=True))
    return 0


def _handle_articulation_project_graph_apply(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import project_joint_graph_apply_result

    return _print_projector_publication(
        project_joint_graph_apply_result(
            args.run_dir,
            output_dir=args.output_dir,
        )
    )


def _handle_articulation_project_gate3a(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import project_joint_gate3a_result

    return _print_projector_publication(
        project_joint_gate3a_result(
            source_path=args.source,
            output_path=args.output,
            report_path=args.report,
            closeout_path=args.closeout,
            run_plan_path=args.run_plan,
            intake_path=args.intake,
            authoring_receipt_path=args.authoring_receipt,
            output_dir=args.output_dir,
        )
    )


def _handle_articulation_project_gate3b(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import project_joint_gate3b_result

    return _print_projector_publication(
        project_joint_gate3b_result(
            source_path=args.source,
            output_path=args.output,
            report_path=args.report,
            closeout_path=args.closeout,
            run_plan_path=args.run_plan,
            intake_path=args.intake,
            authoring_receipt_path=args.authoring_receipt,
            output_dir=args.output_dir,
        )
    )


def _handle_articulation_project_dynamic(args: argparse.Namespace) -> int:
    from content_agent_workflows.articulation import project_joint_dynamic_result

    return _print_projector_publication(
        project_joint_dynamic_result(
            source_path=args.source,
            output_path=args.output,
            receipt_path=args.receipt,
            profile_id=args.profile_id,
            artifact_map_path=args.artifact_map,
            output_dir=args.output_dir,
        )
    )


__all__ = ["add_articulation_capability_subcommands"]
