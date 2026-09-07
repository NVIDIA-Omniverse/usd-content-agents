# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper orchestration for agent-managed external-runtime (BYOR) refinement.

This is the agentic counterpart of ``physics-agent refine-external``. The
engine keeps qualification, fixed-seed tuning, recording integrity, and
winner rendering; the built-in VLM judge and LLM refiner are replaced by a
wrapper-launched coding agent that reviews rendered evidence and writes a
digest-bound decision chain, mirroring the in-house agent-owned tuning phase.

Phases:

1. **Qualification (no approval digest).** Run the engine's qualification
   into ``<run_dir>/qualification``, print the digest, and stop. The operator
   reviews ``qualification.json`` plus its PNG frame sequence and re-invokes with
   ``--approve-qualification sha256:<digest>``. The child agent never
   approves its own qualification.
2. **Agent session (with approval digest).** Validate the approval fail-fast,
   start the ``ExternalTuningBroker``, and launch one child-agent session
   that owns the refine outer loop through the broker's sweep client.
3. **Conclusion.** Verify the decision chain, every cited sweep/evidence/
   frame digest against the broker ledger (rehashing bytes on disk), and —
   only for a verified ``accepted`` result — publish ``final/`` from the
   broker-private engine run directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from content_agent_workflows.physics.external_tuning_contract import (
    PhysicsExternalTuningDecision,
    PhysicsExternalTuningResult,
    load_physics_external_tuning_result,
    sha256_file,
    verify_external_decision_chain,
)
from world_understanding.utils.file_locking import exclusive_descriptor_lock

from content_workflow_cli.external_tuning_broker import (
    DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS,
    DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS,
    ExternalTuningBroker,
)
from content_workflow_cli.prompts import build_physics_external_refine_prompt
from content_workflow_cli.runner import (
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CODEX,
    UnsafeRunArtifactError,
    _create_private_raw_dir,
    _reject_unsafe_run_links,
    _run_child_agent,
)
from content_workflow_cli.trace import TraceWriter

logger = logging.getLogger(__name__)

# Engine-tune content in the published final bundle is entirely
# digest-driven from the broker record: core tune files (run_spec.json,
# best_params.json — CORE_TUNE_ARTIFACT_NAMES) and declared winner outputs
# (spec.publish_artifacts under tune/outputs) republish against digests
# pinned at sweep success. Trial subprocess logs and raw request payloads
# stay diagnostic in the broker workspace, mirroring the engine refine
# loop's own final/ policy. The engine's history.jsonl is deliberately
# absent: its rows carry broker-private replica paths, so final/history.jsonl
# is regenerated from the verified evidence packet's portable history.
# Render evidence is published from the broker's digest-verified evidence
# copies, never from the engine's mutable tune directory (see
# _publish_final).


def _bundle_relative_tune_results(
    value: Any, *, tune_dir: Path, private_root: Path, final_dir: Path
) -> Any:
    """Rewrite engine-run absolute paths into bundle-relative ones.

    The raw ``external_tune_results.json`` serializes qualification/approval
    artifacts and trial recordings as absolute broker-private paths; those
    targets are deleted when a validated run releases the workspace, so the
    published copy must reference the bundle's own files (or ``None`` for
    artifacts the bundle deliberately does not carry) to stay a portable
    terminal snapshot.
    """

    if isinstance(value, dict):
        return {
            key: _bundle_relative_tune_results(
                item,
                tune_dir=tune_dir,
                private_root=private_root,
                final_dir=final_dir,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _bundle_relative_tune_results(
                item,
                tune_dir=tune_dir,
                private_root=private_root,
                final_dir=final_dir,
            )
            for item in value
        ]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        try:
            path.relative_to(private_root)
        except ValueError:
            return value
        try:
            relative = path.relative_to(tune_dir)
        except ValueError:
            return None
        if (final_dir / relative).is_file():
            return relative.as_posix()
        return None
    if isinstance(value, str) and len(Path(value).parts) > 1:
        # The engine also serializes output-local paths as relative strings
        # (per-trial dirs, replica artifacts, scored recordings). A relative
        # string is treated as a tune-dir path only when it actually resolves
        # there, so ordinary prose never matches; references the bundle does
        # not carry are nulled rather than left dangling.
        relative = Path(value)
        candidate = tune_dir / relative
        if candidate.exists():
            try:
                confined_relative = candidate.resolve(strict=True).relative_to(
                    tune_dir.resolve(strict=True)
                )
            except (OSError, ValueError):
                return None
            return (
                confined_relative.as_posix()
                if (final_dir / confined_relative).is_file()
                else None
            )
    return value


@dataclass(frozen=True)
class PhysicsExternalRefineConfig:
    """Config for one agent-managed external refinement run."""

    child_launch_profile: ClassVar[str] = "physics.refine_external.agentic"
    repo_root: Path
    runtime_config: Path
    user_prompt: str
    output_dir: Path
    approval_digest: str | None = None
    reference_images: list[Path] = field(default_factory=list)
    max_iterations: int = DEFAULT_EXTERNAL_REFINE_MAX_ITERATIONS
    max_trials: int = 6
    sweep_deadline_seconds: float = DEFAULT_EXTERNAL_SWEEP_DEADLINE_SECONDS
    phase_deadline_seconds: float | None = None
    additional_instructions: str | None = None
    # AgentRuntimeConfig surface consumed by _run_child_agent and the
    # bridges. This workflow has no USD input; usd_path points at the runtime
    # config for skill-staging path logic.
    runner: str = RUNNER_CODEX
    model: str | None = None
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK
    # Must comfortably exceed max_iterations x sweep_deadline_seconds; the
    # default covers the default budget (5 x 7200 s) plus review slack. The
    # CLI derives this from the actual budget when --child-timeout is omitted.
    child_timeout_seconds: float = 39600.0
    agent_cwd: Path | None = None
    reference_files: list[Path] | None = None

    @property
    def usd_path(self) -> Path:
        return self.runtime_config


@dataclass(frozen=True)
class ExternalRefineRunResult:
    run_dir: Path
    status: str
    validated: bool
    returncode: int
    qualification_digest: str | None = None
    final_dir: Path | None = None
    reasons: tuple[str, ...] = ()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _external_sweep_client_path() -> str:
    """Absolute path to the agent-facing external sweep client script."""

    candidate = Path(sys.executable).with_name(
        "content-workflow-physics-external-sweep"
    )
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("content-workflow-physics-external-sweep")
    return found or "content-workflow-physics-external-sweep"


def _load_spec(config: PhysicsExternalRefineConfig) -> Any:
    from physics_agent.tuning.external import load_external_tune_spec

    return load_external_tune_spec(config.runtime_config)


def _run_qualification(
    config: PhysicsExternalRefineConfig,
    spec: Any,
    run_dir: Path,
    trace: TraceWriter,
    tune_runner: Callable[[Any], Any] | None,
) -> ExternalRefineRunResult:
    from physics_agent.tuning.external import ExternalTuneInput, run_external_tune

    runner = tune_runner or run_external_tune
    qualification_dir = run_dir / "qualification"
    result = runner(ExternalTuneInput(config=spec, output_dir=qualification_dir))
    status = str(getattr(result, "status", "") or "failed")
    digest = getattr(result, "qualification_digest", None)
    summary = {
        "status": status,
        "qualification_digest": digest,
        "qualification_path": str(getattr(result, "qualification_path", "") or ""),
        "error": getattr(result, "error", None),
        "artifacts": {
            str(name): str(path)
            for name, path in (getattr(result, "artifacts", {}) or {}).items()
        },
    }
    _write_json(run_dir / "raw" / "external_qualification.json", summary)
    awaiting = status == "awaiting_approval"
    trace.write(
        "external_refine.qualification",
        phase="qualification",
        summary=(
            f"qualification digest {digest}"
            if awaiting
            else f"qualification ended with status {status}"
        ),
        data=summary,
    )
    if awaiting:
        print(
            "External runtime qualification succeeded and is awaiting approval.\n"
            f"  Review: {summary['qualification_path']}\n"
            f"  Digest: {digest}\n"
            "Re-run the same command with:\n"
            f"  --approve-qualification {digest}"
        )
    return ExternalRefineRunResult(
        run_dir=run_dir,
        status=status,
        validated=False,
        returncode=0 if awaiting else 1,
        qualification_digest=digest,
    )


def _validated_reference_image(source: Path) -> Path:
    """Validate through the Physics-owned public reference-image allowlist."""
    from physics_agent.tuning.visual_evidence import validate_reference_image_path

    return validate_reference_image_path(source)


def _stage_reference_media(
    config: PhysicsExternalRefineConfig, run_dir: Path
) -> tuple[list[Path], dict[Path, str]]:
    """Copy operator reference media into the child-readable run directory.

    Each staged copy is hashed immediately so the wrapper can verify after
    the child session that the comparison inputs the review judged are the
    exact bytes the operator supplied — the run directory is child-writable,
    and unbound references would let a child rewrite them before judging.
    The digests live in wrapper memory (and in the published bundle), never
    trusted back from disk.
    """

    staged_images: list[Path] = []
    staged_digests: dict[Path, str] = {}
    media_dir = run_dir / "reference_media"
    for index, source in enumerate(config.reference_images, start=1):
        source = _validated_reference_image(Path(source))
        media_dir.mkdir(parents=True, exist_ok=True)
        # Index-prefix so two operator paths sharing a basename cannot silently
        # overwrite each other when staged.
        destination = media_dir / f"image_{index:02d}_{source.name}"
        shutil.copyfile(source, destination)
        staged_images.append(destination)
        staged_digests[destination] = sha256_file(destination)
    return staged_images, staged_digests


def _verify_reference_media(staged_digests: dict[Path, str]) -> list[str]:
    """Rehash staged reference media against the pre-session digests."""

    failures: list[str] = []
    for staged, expected in staged_digests.items():
        if staged.is_symlink() or not staged.is_file():
            failures.append(f"reference media {staged} is no longer a regular file")
            continue
        observed = sha256_file(staged)
        if observed != expected:
            failures.append(
                f"reference media {staged} changed during the child session: "
                f"staged {expected}, observed {observed}"
            )
    return failures


def _decision_chain_paths(raw_dir: Path) -> list[Path]:
    """Collect contiguous decision files 1..N; reject gaps and strays."""

    paths: list[Path] = []
    index = 1
    while True:
        candidate = raw_dir / f"physics_external_tuning_decision_{index}.json"
        if not candidate.is_file():
            break
        paths.append(candidate)
        index += 1
    strays = sorted(
        path
        for path in raw_dir.glob("physics_external_tuning_decision_*.json")
        if path not in paths
    )
    if strays:
        raise ValueError(
            "non-contiguous decision files: " + ", ".join(str(p) for p in strays)
        )
    return paths


def _verify_and_conclude(
    *,
    broker: ExternalTuningBroker,
    run_dir: Path,
    child_returncode: int,
) -> tuple[str, bool, list[str], PhysicsExternalTuningResult | None, list[str] | None]:
    """Verify the agent's decision chain and result against the broker ledger.

    Returns ``(status, validated, reasons, result, decision_digests)``;
    ``decision_digests`` (accepted results only) are the verified digests of
    the canonical chain files, in order, for digest-bound publication.
    Any verification failure is a ``tool_failure`` — the artifacts cannot be
    trusted, regardless of what the agent claimed.
    """

    raw_dir = run_dir / "raw"
    reasons: list[str] = []
    result_path = raw_dir / "physics_external_tuning_result.json"

    try:
        decision_paths = _decision_chain_paths(raw_dir)
        decisions = verify_external_decision_chain(list(decision_paths))
    except ValueError as exc:
        return "tool_failure", False, [f"decision chain invalid: {exc}"], None, None

    if not result_path.is_file():
        if child_returncode != 0:
            reasons.append(
                f"child agent exited with {child_returncode} and wrote no "
                "result artifact"
            )
            return "tool_failure", False, reasons, None, None
        reasons.append("child agent wrote no result artifact")
        return "unresolved", False, reasons, None, None
    try:
        result = load_physics_external_tuning_result(result_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return "tool_failure", False, [f"result artifact invalid: {exc}"], None, None

    # A failed child invocation (timeout, crash, nonzero exit) is terminal
    # regardless of what it left on disk: a result written before the
    # session died may predate work the session was still doing, so it must
    # not be verified and published as if the session had concluded cleanly.
    if child_returncode != 0:
        return (
            "tool_failure",
            False,
            [
                f"child agent exited with {child_returncode}; a result from "
                "a failed session cannot be trusted"
            ],
            result,
            None,
        )

    # Every broker-referencing decision must verify against the ledger.
    ledger = broker.ledger()
    records_by_iteration = {record["iteration"]: record for record in ledger.values()}
    for decision in decisions:
        if decision.sweep_id is None:
            continue
        ok, reason = broker.verify_claim(
            {
                "sweep_id": decision.sweep_id,
                "evidence_sha256": decision.evidence_sha256,
            },
            # The contract keeps evidence_sha256 optional for stop and
            # revise_search decisions (a failed sweep never publishes
            # evidence, and the model already requires the digest whenever
            # the decision cites an evidence_path); an omitted digest on an
            # intermediate or stopping decision must not convert a completed
            # run into tool_failure at conclusion. A present digest is still
            # fully verified, and publication stays gated on the accept's
            # strict binding.
            allow_missing_evidence=decision.decision in {"stop", "revise_search"},
        )
        if not ok:
            return (
                "tool_failure",
                False,
                [f"decision {decision.iteration} failed verification: {reason}"],
                result,
                None,
            )
        cited = ledger.get(decision.sweep_id)
        if cited is not None and cited["iteration"] != decision.iteration:
            return (
                "tool_failure",
                False,
                [
                    f"decision {decision.iteration} cites sweep "
                    f"{decision.sweep_id} from broker iteration "
                    f"{cited['iteration']}; decisions must judge their own "
                    "iteration's sweep"
                ],
                result,
                None,
            )
        if decision.decision == "revise_search":
            # The declared next search must be the one the following sweep
            # actually ran; otherwise the chain narrates search A while the
            # broker executed search B.
            following = records_by_iteration.get(decision.iteration + 1)
            if (
                following is not None
                and following["active_search"] != decision.next_active_search
            ):
                return (
                    "tool_failure",
                    False,
                    [
                        f"decision {decision.iteration} declared "
                        "next_active_search does not match the search the "
                        f"following sweep ran: declared "
                        f"{decision.next_active_search}, ran "
                        f"{following['active_search']}"
                    ],
                    result,
                    None,
                )

    if result.final_decision_sha256 and decision_paths:
        actual = sha256_file(decision_paths[-1])
        if result.final_decision_sha256 != actual:
            return (
                "tool_failure",
                False,
                [
                    "result.final_decision_sha256 does not match the last "
                    f"decision file: claimed {result.final_decision_sha256}, "
                    f"actual {actual}"
                ],
                result,
                None,
            )

    if result.status != "accepted" and not ledger:
        # A zero-sweep refusal (the child stopped before reserving any
        # sweep) has no executed work to bind; the bare status is an honest
        # terminal answer.
        return result.status, False, reasons, result, None

    # Any result that summarizes executed sweeps — accepted or not — must be
    # verifiably bound to the canonical chain: a mandatory final-decision
    # digest and an exact decision-path match, so a published result cannot
    # claim a chain other than the verified one, and a completed sweep
    # cannot disappear behind a bare terminal status.
    if not result.final_decision_sha256:
        return (
            "tool_failure",
            False,
            ["results that summarize executed sweeps must bind final_decision_sha256"],
            result,
            None,
        )
    # Path-normalized comparison: the agent may legitimately spell the same
    # files relative to the run directory; only a genuinely different chain
    # (different files or order) may void a fully verified result. An empty
    # list is rejected too — the published result must identify the exact
    # chain it summarizes.
    if not result.decision_paths:
        return (
            "tool_failure",
            False,
            ["results that summarize executed sweeps must enumerate decision_paths"],
            result,
            None,
        )
    canonical_paths = [Path(path).resolve() for path in decision_paths]
    claimed_paths = [
        (Path(p) if Path(p).is_absolute() else run_dir / p).resolve()
        for p in result.decision_paths
    ]
    if claimed_paths != canonical_paths:
        return (
            "tool_failure",
            False,
            [
                "result.decision_paths does not match the canonical verified "
                f"chain: claimed {result.decision_paths}, verified "
                f"{[str(path) for path in canonical_paths]}"
            ],
            result,
            None,
        )

    # Every reserved sweep must be accounted for in the decision chain: an
    # accept (or honest stop) for sweep 1 must not silently conclude while
    # an unaccounted sweep 2 was cancelled by shutdown (or is still
    # executing).
    cited_sweeps = {decision.sweep_id for decision in decisions if decision.sweep_id}
    unaccounted = sorted(
        sweep_id for sweep_id in ledger if sweep_id not in cited_sweeps
    )
    if unaccounted:
        return (
            "tool_failure",
            False,
            [
                "results must account for every reserved sweep in the "
                f"decision chain; unaccounted: {unaccounted}"
            ],
            result,
            None,
        )

    # The claimed terminal status must agree with the chain's terminal
    # action (the chain verifier already guarantees accept/stop only occur
    # as the final decision): a result must not report a concluded loop
    # differently from the decision that concluded it. A final stop may
    # honestly conclude as budget_exhausted too — the skill and prompt
    # direct the agent to stop with a rationale when a reservation is
    # refused, and that refusal is budget exhaustion, not a plain stop.
    final_action = decisions[-1].decision if decisions else None
    allowed_statuses = {
        "accept": {"accepted"},
        "stop": {"stopped", "budget_exhausted"},
    }.get(final_action or "")
    if allowed_statuses is not None and result.status not in allowed_statuses:
        return (
            "tool_failure",
            False,
            [
                f"the decision chain ends in {final_action} but the result "
                f"claims status {result.status!r}; expected one of "
                f"{sorted(allowed_statuses)}"
            ],
            result,
            None,
        )

    if result.status != "accepted":
        return result.status, False, reasons, result, None

    # Accepted: the chain must END with a matching, fully verified accept.
    if not decisions or decisions[-1].decision != "accept":
        return (
            "tool_failure",
            False,
            ["result claims accepted but the decision chain does not end in accept"],
            result,
            None,
        )
    accept = decisions[-1]
    selected = result.selected
    if selected is None:
        # The result model validator guarantees this; an explicit check (not
        # an assert, which compiles out under python -O) keeps a tampered
        # artifact on the clean tool_failure path.
        return (
            "tool_failure",
            False,
            ["result claims accepted but carries no selected result"],
            result,
            None,
        )
    failure = _verify_accept(broker, accept, selected)
    if failure is not None:
        return "tool_failure", False, [failure], result, None
    # The digest of every chain file was established during verification:
    # each file is pinned by its successor's prior_decision_sha256 and the
    # last by the result's (already matched) final_decision_sha256.
    # Publication rechecks the archived copies against exactly these values.
    decision_digests = [
        decision.prior_decision_sha256 for decision in decisions[1:]
    ] + [result.final_decision_sha256]
    return "accepted", True, reasons, result, decision_digests


def _verify_accept(
    broker: ExternalTuningBroker,
    accept: PhysicsExternalTuningDecision,
    selected: Any,
) -> str | None:
    """Return a failure reason when the accept does not match broker truth."""

    if accept.selected is None or accept.sweep_id != selected.sweep_id:
        return "result.selected does not match the accepting decision's sweep"
    if accept.selected.model_dump() != selected.model_dump():
        return "result.selected does not match the accepting decision's selection"
    ok, reason = broker.verify_reviewed_frames(
        selected.sweep_id,
        [frame.model_dump() for frame in accept.reviewed_frames],
    )
    if not ok:
        return f"reviewed frames failed verification: {reason}"
    record = broker.ledger().get(selected.sweep_id)
    if record is None:
        return f"sweep {selected.sweep_id!r} has no broker record"
    if record["status"] != "succeeded":
        return (
            f"accepted sweep {selected.sweep_id} is {record['status']}, not succeeded"
        )
    broker_params = {
        name: float(value) for name, value in record["best_params"].items()
    }
    claimed_params = {
        name: float(value) for name, value in selected.best_params.items()
    }
    if broker_params != claimed_params:
        return (
            "selected best_params do not match the broker record: "
            f"claimed {claimed_params}, broker recorded {broker_params}"
        )
    if (record.get("recording_sha256") or None) != selected.recording_sha256:
        return (
            "selected recording_sha256 does not match the broker record: "
            f"claimed {selected.recording_sha256}, broker recorded "
            f"{record.get('recording_sha256')}"
        )
    return None


def _publish_final(
    *,
    broker: ExternalTuningBroker,
    run_dir: Path,
    result: PhysicsExternalTuningResult,
    decision_paths: list[Path],
    decision_digests: list[str],
    reference_media: dict[Path, str] | None = None,
    contract_path: Path | None = None,
    contract_sha256: str | None = None,
) -> Path:
    """Publish the accepted result from broker-verified artifacts."""

    from content_workflow_cli.tuning_broker import _reject_child_output_links

    if result.selected is None:
        # Verified before publication; explicit (not an assert, which
        # compiles out under python -O) so a tampered artifact fails closed.
        raise ValueError("accepted result carries no selected result")
    sweep_id = result.selected.sweep_id
    tune_dir = broker.sweep_run_dir(sweep_id)
    final_dir = _reject_child_output_links(run_dir, run_dir / "final")
    if final_dir.exists():
        # Child-authored content at this path is not trusted output.
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)

    manifest: dict[str, str] = {}

    def _publish_file(source: Path, destination: Path) -> None:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"final artifact is not a regular file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[destination.relative_to(final_dir).as_posix()] = sha256_file(
            destination
        )

    def _publish_verified(
        raw: Any, destination: Path, expected: Any, label: str
    ) -> None:
        # Evidence is published from digest-verified copies, never from
        # the engine's mutable private directory: that directory can
        # change or disappear after evidence publication (e.g. a
        # lingering BYOR subprocess), and the final bundle must carry
        # exactly the files whose digests verification checked. Every
        # copy is rehashed against the recorded digest and publication
        # fails closed on any mismatch.
        if not raw or not Path(raw).is_file():
            raise ValueError(
                f"accepted sweep {sweep_id} has no verified {label} on disk to publish"
            )
        _publish_file(Path(raw), destination)
        published = manifest[destination.relative_to(final_dir).as_posix()]
        if not expected or published != expected:
            raise ValueError(
                f"published {label} digest does not match the verified "
                f"record: published {published}, recorded {expected}"
            )

    try:
        record = broker.ledger()[sweep_id]
        # Core tune artifacts republish against the digests the broker
        # pinned when the sweep succeeded: a post-sweep mutation of the
        # engine's tune directory (e.g. a lingering BYOR subprocess
        # rewriting best_params.json) fails closed instead of silently
        # shipping changed bytes or omitting the file.
        for name, digest in (record.get("tune_artifacts") or {}).items():
            if name == "external_tune_results.json":
                # Republished below as a portable rewrite (its raw form
                # carries broker-private absolute paths), after the raw
                # bytes verify against this same pinned digest.
                continue
            _publish_verified(
                tune_dir / name, final_dir / name, digest, f"tune artifact {name}"
            )
        # Declared winner outputs republish against the digests the engine
        # pinned when it promoted them: every declared output is required,
        # so a lingering BYOR descendant that changes or removes files
        # under tune/outputs after the sweep succeeds fails closed instead
        # of the bundle silently carrying whatever currently exists.
        for relative, digest in (record.get("published_outputs") or {}).items():
            _publish_verified(
                tune_dir / relative,
                final_dir / relative,
                digest,
                f"declared output {relative}",
            )
        # The decision chain is republished against the exact digests the
        # verification pass established (each file's digest is pinned by its
        # successor's prior_decision_sha256 and the result's
        # final_decision_sha256), so the archived chain cannot drift from
        # the verified one.
        decisions_dir = final_dir / "decisions"
        for path, digest in zip(decision_paths, decision_digests, strict=True):
            _publish_verified(
                path, decisions_dir / path.name, digest, f"decision {path.name}"
            )
        result_source = run_dir / "raw" / "physics_external_tuning_result.json"
        _publish_file(result_source, final_dir / "result.json")
        if contract_path is not None:
            # The exact acceptance criterion the review was judged against,
            # verified against the pre-session digest, so a rerun that
            # overwrites raw/ never strips an archived accepted deliverable
            # of its contract.
            _publish_verified(
                contract_path,
                final_dir / "physics_external_contract.json",
                contract_sha256,
                "operator contract",
            )
        # Mandatory: an accepted sweep always published a verified evidence
        # packet; a missing or drifted source here means the deliverable is
        # incomplete and publication must fail closed rather than silently
        # omit or swap it.
        _publish_verified(
            record.get("evidence_path"),
            final_dir / "evidence.json",
            record.get("evidence_sha256"),
            "evidence packet",
        )
        # The published history is derived from the digest-verified evidence
        # packet rather than copied from the engine's history.jsonl: the raw
        # file serializes broker-private replica trial_dir/artifact paths
        # that dangle once a validated run releases the workspace.
        history_entries = json.loads(
            (final_dir / "evidence.json").read_text(encoding="utf-8")
        ).get("history", [])
        history_destination = final_dir / "history.jsonl"
        history_destination.write_text(
            "".join(json.dumps(entry) + "\n" for entry in history_entries),
            encoding="utf-8",
        )
        manifest["history.jsonl"] = sha256_file(history_destination)
        # The recording is published from the broker's already-verified copy
        # and rehashed against the accepted digest: the private tune artifact
        # can change or disappear after evidence publication (e.g. a
        # lingering BYOR subprocess), and the final recording must be exactly
        # the one that produced the reviewed frames.
        recording_raw = record.get("recording_path")
        if not recording_raw or not Path(recording_raw).is_file():
            raise ValueError(
                f"accepted sweep {sweep_id} has no verified recording on "
                "disk to publish"
            )
        _publish_file(Path(recording_raw), final_dir / "best_recording.usd")
        # The contract requires accepted results to bind the recording
        # digest, so this comparison is unconditional (a None claim could
        # never verify against the broker record anyway).
        expected_recording = result.selected.recording_sha256
        if manifest["best_recording.usd"] != expected_recording:
            raise ValueError(
                "published recording digest does not match the accepted "
                f"recording: published {manifest['best_recording.usd']}, "
                f"accepted {expected_recording}"
            )

        frames = record.get("frames") or []
        if not frames:
            raise ValueError(
                f"accepted sweep {sweep_id} recorded no verified render "
                "frames to publish"
            )
        for frame in frames:
            frame_raw = frame.get("path")
            frame_name = Path(str(frame_raw)).name if frame_raw else "frame"
            _publish_verified(
                frame_raw,
                final_dir / "render" / frame_name,
                frame.get("sha256"),
                f"render frame {frame_name}",
            )
        _publish_verified(
            record.get("response_metadata_path"),
            final_dir / "render" / "render_response_metadata.json",
            record.get("response_metadata_sha256"),
            "render response metadata",
        )
        # The exact comparison inputs the review judged, verified against
        # the digests taken when they were staged, so an accepted bundle
        # reproduces which references were reviewed.
        for staged, expected in (reference_media or {}).items():
            _publish_verified(
                staged,
                final_dir / "reference_media" / staged.name,
                expected,
                f"reference media {staged.name}",
            )
        # The terminal tune result is documented output: it is required, and
        # the portable rewrite starts from raw bytes verified against the
        # digest the broker pinned at sweep success, so a post-sweep rewrite
        # or deletion fails closed instead of packaging changed metadata.
        results_source = tune_dir / "external_tune_results.json"
        expected_results = (record.get("tune_artifacts") or {}).get(
            "external_tune_results.json"
        )
        if not expected_results or not results_source.is_file():
            raise ValueError(
                f"accepted sweep {sweep_id} has no verified "
                "external_tune_results.json to publish"
            )
        raw_results = results_source.read_bytes()
        observed_results = hashlib.sha256(raw_results).hexdigest()
        if observed_results != expected_results:
            raise ValueError(
                "external_tune_results.json does not match the digest pinned "
                f"at sweep success: observed {observed_results}, recorded "
                f"{expected_results}"
            )
        portable = _bundle_relative_tune_results(
            json.loads(raw_results.decode("utf-8")),
            tune_dir=tune_dir,
            private_root=broker.private_dir,
            final_dir=final_dir,
        )
        results_destination = final_dir / "external_tune_results.json"
        results_destination.write_text(json.dumps(portable, indent=2), encoding="utf-8")
        manifest["external_tune_results.json"] = sha256_file(results_destination)
        # Portable path index: the digest-bound result.json/evidence.json
        # copies must stay byte-identical to the verified originals, so
        # their absolute run/broker references are mapped here to the
        # bundle's own files instead (matched by digest; None when the
        # bundle deliberately does not carry the target). A later rerun
        # wipes the referenced originals, so the archived bundle stays
        # self-describing through this index.
        digest_to_bundle: dict[str, str] = {}
        for published in sorted(final_dir.rglob("*")):
            if published.is_file():
                digest_to_bundle.setdefault(
                    sha256_file(published),
                    published.relative_to(final_dir).as_posix(),
                )
        path_index: dict[str, str | None] = {
            str(path): f"decisions/{path.name}" for path in decision_paths
        }
        evidence_payload = json.loads(
            (final_dir / "evidence.json").read_text(encoding="utf-8")
        )
        for frame in evidence_payload.get("frames", []):
            # Each frame maps to its own published render/ copy: settled
            # rollouts produce byte-identical frames, and collapsing them
            # through digest_to_bundle would resolve distinct chronological
            # source paths to one bundle file.
            frame_copy = f"render/{Path(str(frame.get('path'))).name}"
            path_index[str(frame.get("path"))] = (
                frame_copy
                if manifest.get(frame_copy) == str(frame.get("sha256"))
                else digest_to_bundle.get(str(frame.get("sha256")))
            )
        if evidence_payload.get("recording_path"):
            path_index[str(evidence_payload["recording_path"])] = digest_to_bundle.get(
                str(evidence_payload.get("recording_sha256"))
            )
        # The engine's selected-evidence descriptors reference the scored
        # (and published) recording by broker-private path; both carry the
        # same bytes as the published best_recording.usd, so index them by
        # digest rather than leaving the references dangling after the
        # workspace is released.
        selected_evidence = evidence_payload.get("selected_evidence") or {}
        for descriptor_name in ("scored_recording", "published_recording"):
            descriptor = selected_evidence.get(descriptor_name)
            if isinstance(descriptor, dict) and descriptor.get("path"):
                descriptor_digest = str(descriptor.get("sha256") or "").removeprefix(
                    "sha256:"
                )
                path_index.setdefault(
                    str(descriptor["path"]), digest_to_bundle.get(descriptor_digest)
                )
        provenance = evidence_payload.get("render_provenance") or {}
        if provenance.get("response_metadata_path"):
            path_index[str(provenance["response_metadata_path"])] = (
                digest_to_bundle.get(str(provenance.get("response_metadata_sha256")))
            )
        # Each copied decision cites the broker's absolute evidence_path;
        # map it (and the relative decision-path spellings validation
        # permits) so the archived bundle resolves every reference it
        # carries.
        for path in decision_paths:
            decision_payload = json.loads(path.read_text(encoding="utf-8"))
            cited_evidence = decision_payload.get("evidence_path")
            if cited_evidence:
                path_index.setdefault(
                    str(cited_evidence),
                    digest_to_bundle.get(str(decision_payload.get("evidence_sha256"))),
                )
            try:
                relative_spelling = path.relative_to(run_dir)
            except ValueError:
                continue
            path_index.setdefault(
                relative_spelling.as_posix(), f"decisions/{path.name}"
            )
        _write_json(
            final_dir / "portable_paths.json",
            {
                "schema_version": ("content-agents.physics-external-portable-paths.v1"),
                "paths": path_index,
            },
        )
        manifest["portable_paths.json"] = sha256_file(final_dir / "portable_paths.json")
        _write_json(final_dir / "manifest.json", manifest)
    except BaseException:
        # A partial final/ without manifest.json must not survive a failed
        # publication: final/ always corresponds to the current conclusion.
        shutil.rmtree(final_dir, ignore_errors=True)
        raise
    return final_dir


def _clear_stale_run_artifacts(run_dir: Path) -> None:
    """Archive the prior final/ and remove stale tuning artifacts.

    Re-invoking the same ``--output-dir`` is the documented workflow
    (qualification pass, then approval pass), but a rerun must not inherit a
    previous run's promotable ``final/`` bundle, decision chain, or iteration
    dirs: a shorter rerun would fail chain verification on stale
    higher-numbered decisions, and an honest stop would leave a stale
    accepted bundle next to a conclusion that reports no publication. The
    prior ``final/`` is archived to ``<run_dir>.final.<n>/`` beside the run
    directory — never deleted, and outside the child-writable workspace —
    and the qualification directory is retained; the approval phase
    validates against it.
    """

    final_dir = run_dir / "final"
    if final_dir.is_symlink():
        final_dir.unlink()
    elif final_dir.exists():
        # Archive rather than delete: after a validated run released the
        # broker workspace, final/ is the only surviving copy of the accepted
        # deliverable, and a rerun that concludes non-accepted (honest stop,
        # tool_failure) must not have destroyed it. The stale decision and
        # tuning artifacts below still must go — they would break chain
        # verification — so the bundle moves aside. The archive lives BESIDE
        # the run directory, outside the child's workspace-write sandbox,
        # so the new session cannot corrupt or delete it.
        archive_index = 1
        while (run_dir.parent / f"{run_dir.name}.final.{archive_index}").exists():
            archive_index += 1
        archived = run_dir.parent / f"{run_dir.name}.final.{archive_index}"
        final_dir.rename(archived)
        logger.info("archived the previous final/ deliverable to %s", archived)
    tuning_dir = run_dir / "tuning"
    if tuning_dir.is_symlink():
        tuning_dir.unlink()
    elif tuning_dir.exists():
        shutil.rmtree(tuning_dir)
    # Staged reference media are restaged from the current invocation's
    # arguments; a rerun with fewer or different references must not leave
    # the previous run's files visible to the child — they would be absent
    # from the current digest set and final bundle, so a review could rest
    # on unbound evidence.
    media_dir = run_dir / "reference_media"
    if media_dir.is_symlink():
        media_dir.unlink()
    elif media_dir.exists():
        shutil.rmtree(media_dir)
    raw_dir = run_dir / "raw"
    if raw_dir.is_dir():
        stale = [
            *raw_dir.glob("physics_external_tuning_decision_*.json"),
            raw_dir / "physics_external_tuning_result.json",
            raw_dir / "external_refine_conclusion.json",
            raw_dir / "external_sweep_ledger.jsonl",
        ]
        for path in stale:
            if path.is_file() or path.is_symlink():
                path.unlink()


def run_physics_external_refine(
    config: PhysicsExternalRefineConfig,
    *,
    tune_runner: Callable[[Any], Any] | None = None,
    child_agent_runner: Callable[..., int] | None = None,
) -> ExternalRefineRunResult:
    """Run one agent-managed external refinement phase end to end.

    ``tune_runner`` and ``child_agent_runner`` are injectable for tests;
    production uses ``run_external_tune`` and the wrapper's child bridges.

    The whole qualification/session/conclusion sequence holds an exclusive
    per-output-dir lock: two concurrent invocations sharing an
    ``--output-dir`` would wipe each other's artifacts and race on the same
    result filenames. The lock file lives *beside* the run directory, outside
    the child sandbox, so a child cannot tamper with it.
    """

    # The behavior goal is the acceptance criterion the child's visual
    # review judges against; a blank one would let the workflow accept and
    # publish a result with no meaningful criterion at all.
    if not config.user_prompt.strip():
        raise ValueError(
            "user_prompt must contain a non-empty behavior goal; got a "
            "blank or whitespace-only value"
        )

    run_dir = config.output_dir.expanduser().resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.parent / f".{run_dir.name}.external-refine.lock"
    with lock_path.open("a+b") as lock_file:
        lock = exclusive_descriptor_lock(lock_file.fileno())
        try:
            lock.__enter__()
        except BlockingIOError as exc:
            raise RuntimeError(
                "another physics refine-external run is already active for "
                f"{run_dir}; concurrent runs sharing an output directory "
                "would corrupt each other's artifacts"
            ) from exc
        try:
            return _run_physics_external_refine_locked(
                config,
                run_dir=run_dir,
                tune_runner=tune_runner,
                child_agent_runner=child_agent_runner,
            )
        finally:
            lock.__exit__(None, None, None)


def _run_physics_external_refine_locked(
    config: PhysicsExternalRefineConfig,
    *,
    run_dir: Path,
    tune_runner: Callable[[Any], Any] | None,
    child_agent_runner: Callable[..., int] | None,
) -> ExternalRefineRunResult:
    # Reject links anywhere in a reused run tree before any wrapper write: a
    # child from a previous session could have planted a symlink (e.g. at
    # raw/) that would redirect unsandboxed writes, and the stale-artifact
    # wipe below must never follow one outside the run directory. The
    # operator-supplied outer path is resolved first — the child's sandbox is
    # confined to the run directory, so only the tree contents are
    # child-controlled.
    _reject_unsafe_run_links(run_dir, allow_missing=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _create_private_raw_dir(run_dir)
    trace = TraceWriter(run_dir)
    spec = _load_spec(config)

    # Refuse a non-OVRTX evidence renderer before qualification: the engine
    # accepts playback_renderer "remote", but this workflow's broker rejects
    # unattested render provenance at evidence publication — which would
    # otherwise surface only after a full sweep of customer-simulator wall
    # clock. Fail the operator fast, at config load.
    playback_renderer = getattr(
        getattr(spec, "evidence", None), "playback_renderer", None
    )
    if playback_renderer != "ovrtx":
        raise ValueError(
            f"evidence.playback_renderer {playback_renderer!r} is not "
            "supported by the agent-managed external refine workflow; final "
            "review evidence must render through the OVRTX backend "
            "(set evidence.playback_renderer: ovrtx in the runtime config)"
        )

    if config.approval_digest is None:
        return _run_qualification(config, spec, run_dir, trace, tune_runner)

    from physics_agent.tuning.external import validate_qualification_approval

    qualification_dir = run_dir / "qualification"
    validate_qualification_approval(
        spec,
        qualification_dir=qualification_dir,
        approval_digest=config.approval_digest,
    )

    # Every fail-fast input check must precede the stale wipe: after a
    # validated run releases the broker workspace, final/ is the only copy of
    # the accepted deliverable, and a rerun that fails on a typo'd digest,
    # config, adapter edit, or missing reference-media path must not have
    # destroyed it first. Mirrors the engine loop, which wipes prior
    # iterations only after approval validates.
    for media_source in config.reference_images:
        _validated_reference_image(Path(media_source))
    _clear_stale_run_artifacts(run_dir)

    staged_images, staged_media_digests = _stage_reference_media(config, run_dir)
    # The child bridges run with run_dir as their CWD, so a caller-relative
    # --reference-image path would resolve wrongly there; hand the child the
    # staged absolute copies instead of the operator's original paths.
    config = replace(
        config,
        reference_images=list(staged_images),
    )
    broker = ExternalTuningBroker(
        run_dir=run_dir,
        spec=spec,
        qualification_dir=qualification_dir,
        approval_digest=config.approval_digest,
        max_sweeps=config.max_iterations,
        max_trials_per_sweep=config.max_trials,
        sweep_deadline_seconds=config.sweep_deadline_seconds,
        phase_deadline_seconds=config.phase_deadline_seconds,
        tune_runner=tune_runner,
    )
    broker.start()
    status = "tool_failure"
    validated = False
    reasons: list[str] = []
    final_dir: Path | None = None
    try:
        contract = {
            "schema_version": "content-agents.physics-external-contract.v1",
            "runtime_config": str(config.runtime_config.resolve()),
            "task": spec.task,
            "behavior_goal": config.user_prompt,
            "objective": {
                "name": spec.objective.name,
                "unit": spec.objective.unit,
                "direction": spec.objective.direction,
            },
            "qualification_digest": config.approval_digest,
            "budget": {
                "max_sweeps": config.max_iterations,
                "max_trials_per_sweep": config.max_trials,
                "sweep_deadline_seconds": config.sweep_deadline_seconds,
                "phase_deadline_seconds": config.phase_deadline_seconds,
            },
        }
        contract_path = run_dir / "raw" / "physics_external_contract.json"
        _write_json(contract_path, contract)
        # The contract is the raw acceptance criterion the review is judged
        # against; it becomes child-writable, so its pre-session bytes are
        # pinned here, re-verified at conclusion, and published with the
        # final bundle.
        contract_digest = sha256_file(contract_path)
        prompt = build_physics_external_refine_prompt(
            run_dir=run_dir,
            broker_url=broker.url,
            sweep_client_path=_external_sweep_client_path(),
            contract_path=contract_path,
            user_prompt=config.user_prompt,
            task_name=spec.task,
            objective=contract["objective"],
            parameter_catalog=[
                {"name": parameter.name, "integer": parameter.integer}
                for parameter in spec.parameter_catalog
            ],
            initial_active_search={
                param.name: {
                    "min": float(param.min_value),
                    "max": float(param.max_value),
                }
                for param in spec.params
            },
            nominal_params={
                name: float(value)
                for name, value in spec.qualification.nominal_params.items()
            },
            max_iterations=config.max_iterations,
            max_trials_per_sweep=config.max_trials,
            sweep_deadline_seconds=config.sweep_deadline_seconds,
            reference_images=staged_images,
            additional_instructions=config.additional_instructions,
        )
        (run_dir / "raw" / "prompt.txt").write_text(prompt, encoding="utf-8")
        trace.write(
            "external_refine.session.started",
            phase="agent_session",
            summary=f"external refine session for task {spec.task}",
            data={"broker_url": broker.url, "budget": contract["budget"]},
        )
        agent_runner = child_agent_runner or _run_child_agent
        try:
            child_returncode = agent_runner(
                config=config,
                prompt=prompt,
                run_dir=run_dir,
                child_output_path=run_dir / "raw" / "child_output.json",
                child_final_path=run_dir / "raw" / "child_final.json",
                # The sandbox allowlist takes hostnames, not URLs; the child
                # gets the full broker endpoint from the prompt instead.
                extra_allowed_hosts=[
                    host for host in [urlparse(broker.url).hostname] if host
                ],
            )
        except UnsafeRunArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - conclude with tool_failure
            # An operational runner failure (timeout, provider startup) is a
            # terminal tool_failure with durable conclusion artifacts, not a
            # bare CLI exception that skips verification entirely.
            logger.exception("external refine child runner failed")
            reasons.append(f"child runner failed: {type(exc).__name__}: {exc}")
            child_returncode = 2
        broker.close()
        status, validated, verify_reasons, result, decision_digests = (
            _verify_and_conclude(
                broker=broker,
                run_dir=run_dir,
                child_returncode=child_returncode,
            )
        )
        reasons.extend(verify_reasons)
        # The staged reference media are the review's comparison inputs; a
        # child that rewrote them judged against different references than
        # the operator supplied, so a mismatch voids the accept.
        media_failures = _verify_reference_media(staged_media_digests)
        if media_failures:
            reasons.extend(media_failures)
            if validated:
                status = "tool_failure"
                validated = False
        # The contract is the raw behavior goal the review was judged
        # against; a child edit here would falsify the acceptance criterion
        # the published result claims to satisfy.
        if (
            contract_path.is_symlink()
            or not contract_path.is_file()
            or sha256_file(contract_path) != contract_digest
        ):
            reasons.append(
                "physics_external_contract.json changed during the child "
                "session; the behavior goal the review was judged against "
                "cannot be trusted"
            )
            if validated:
                status = "tool_failure"
                validated = False
        if validated and result is not None:
            try:
                final_dir = _publish_final(
                    broker=broker,
                    run_dir=run_dir,
                    result=result,
                    decision_paths=_decision_chain_paths(run_dir / "raw"),
                    decision_digests=decision_digests or [],
                    reference_media=staged_media_digests,
                    contract_path=contract_path,
                    contract_sha256=contract_digest,
                )
            except UnsafeRunArtifactError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep recovery artifacts
                # A failed publication must not conclude as accepted: the
                # partial final/ was removed, so downgrading keeps the broker
                # workspace (the only remaining copy) for recovery instead of
                # releasing it under a success conclusion.
                logger.exception("publishing final/ failed")
                status = "tool_failure"
                validated = False
                final_dir = None
                reasons.append(f"publishing final/ failed: {type(exc).__name__}: {exc}")
    finally:
        broker.close()
        # A child that replaced run artifacts with symlinks must not be able
        # to redirect these unsandboxed wrapper writes: if the tree fails the
        # link-safety check, keep the diagnostics in the wrapper-private
        # broker workspace instead of following child-controlled paths.
        try:
            _reject_unsafe_run_links(run_dir)
            diagnostics_dir = run_dir / "raw"
        except UnsafeRunArtifactError as unsafe_exc:
            diagnostics_dir = broker.private_dir / "unsafe_run_diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            logger.error(
                "run directory failed the link-safety check (%s); writing "
                "conclusion diagnostics to %s",
                unsafe_exc,
                diagnostics_dir,
            )
        if broker.ledger_path.is_file():
            shutil.copyfile(
                broker.ledger_path,
                diagnostics_dir / "external_sweep_ledger.jsonl",
            )
        # Release before writing the conclusion so it records what actually
        # happened: a still-alive worker thread (a BYOR runtime that
        # outlived the close() join) makes the release a no-op, and the
        # leaked workspace path must be reported for later cleanup instead
        # of silently recorded as gone. Skipped when conclusion diagnostics
        # were redirected into the broker workspace — releasing it would
        # delete them.
        released = False
        if validated and diagnostics_dir == run_dir / "raw":
            released = broker.release_private_dir()
            if not released:
                logger.warning(
                    "broker workspace release skipped (a worker thread is "
                    "still alive); leaked directory recorded in the "
                    "conclusion: %s",
                    broker.private_dir,
                )
        budget_snapshot = broker.budget_view()
        conclusion = {
            "status": status,
            "validated": validated,
            "reasons": reasons,
            "final_dir": str(final_dir) if final_dir else None,
            # Transparency for the advertised whole-phase ceiling: the
            # deadline gates sweep reservation (the broker refuses new
            # sweeps past it); post-sweep review and authoring time is
            # bounded by child_timeout_seconds instead, so a completed
            # verified run is never voided here — but the overrun, if any,
            # is durably recorded for the operator.
            "phase_deadline_seconds": budget_snapshot["phase_deadline_seconds"],
            "phase_seconds_remaining": budget_snapshot["phase_seconds_remaining"],
            "phase_deadline_exceeded": (
                budget_snapshot["phase_deadline_seconds"] is not None
                and budget_snapshot["phase_seconds_remaining"] == 0.0
            ),
            # Kept on non-validated runs so engine internals (per-trial
            # subprocess logs) remain inspectable; removed on success. On a
            # validated run this is null only when the workspace really was
            # removed — otherwise it records the leaked path.
            "broker_private_dir": (None if released else str(broker.private_dir)),
            "ledger": broker.ledger(),
        }
        _write_json(diagnostics_dir / "external_refine_conclusion.json", conclusion)
        trace.write(
            "external_refine.concluded",
            phase="conclusion",
            summary=f"external refine concluded with status {status}",
            data={k: v for k, v in conclusion.items() if k != "ledger"},
        )
        if not validated:
            logger.info(
                "external refine ended with status %s; broker workspace kept "
                "for diagnosis at %s",
                status,
                broker.private_dir,
            )
    return ExternalRefineRunResult(
        run_dir=run_dir,
        status=status,
        validated=validated,
        returncode=0 if validated else 1,
        qualification_digest=config.approval_digest,
        final_dir=final_dir,
        reasons=tuple(reasons),
    )
