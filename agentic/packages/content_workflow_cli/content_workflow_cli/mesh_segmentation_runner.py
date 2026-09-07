# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh-session launcher for the mesh-segmentation workflow skill."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from array import array
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar
from urllib.parse import urlparse

import numpy as np
from content_agent_workflows.common.memory import (
    AgentMemory,
    MemoryArtifactInput,
    MemoryInteraction,
    MemoryOutcome,
    MemorySearchQuery,
    RememberRequest,
    resolve_memory_root,
)
from content_agent_workflows.mesh_segmentation_contract import (
    MESH_SEGMENTATION_REQUIRED_SKILLS,
    REQUIRED_FINAL_ARTIFACTS,
    REQUIRED_RECOGNITION_FINAL_ARTIFACTS,
    REQUIRED_TARGETED_FINAL_ARTIFACTS,
    required_mesh_segmentation_artifacts,
)
from PIL import Image

from .memory_broker import (
    MEMORY_ORIGIN_AGENT_TAG,
    MEMORY_ORIGIN_LAUNCHER_TAG,
    AgentMemoryBroker,
)
from .mesh_segmentation_vqa import (
    validate_part_lock_selection_evidence,
    validate_targeted_selection_evidence,
)
from .prompts import CONTROLLED_JSON_ARTIFACT_WRITE
from .runner import (
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_DANGER_FULL_ACCESS,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CODEX,
    SUPPORTED_CLAUDE_EXECUTION_MODES,
    SUPPORTED_CODEX_SANDBOX_MODES,
    SUPPORTED_RUNNERS,
    WatchdogFailure,
    _workflow_usd_cli_session_id,
    append_child_runner_error,
    build_codex_sdk_request,
    chmod_private,
    codex_bridge_env,
    combine_watchdogs,
    console_stream,
    effective_codex_home,
    lexical_absolute_path,
    parent_usd_cli_prompt_contract,
    reject_unsafe_run_links,
    run_child_agent,
    run_progress_summary_for_log,
    run_subprocess_with_timeout,
    stage_agent_skills,
    start_parent_usd_cli_capability,
    stop_parent_usd_cli_capability,
    terminal_success_detector_for_bridge,
    write_private_json,
)
from .trace import TraceWriter, UnsafeRunArtifactError, utc_now

_append_child_runner_error = append_child_runner_error
_build_codex_sdk_request = build_codex_sdk_request
_chmod_private = chmod_private
_combine_watchdogs = combine_watchdogs
_codex_bridge_env = codex_bridge_env
_console_stream = console_stream
_effective_codex_home = effective_codex_home
_lexical_absolute_path = lexical_absolute_path
_reject_unsafe_run_links = reject_unsafe_run_links
_run_child_agent = run_child_agent
_run_progress_summary_for_log = run_progress_summary_for_log
_run_subprocess_with_timeout = run_subprocess_with_timeout
_stage_agent_skills = stage_agent_skills
_terminal_success_detector_for_bridge = terminal_success_detector_for_bridge
_write_private_json = write_private_json

LOGGER = logging.getLogger(__name__)

REQUEST_SCHEMA_VERSION = "content-agents.mesh-segmentation-request.v3"
FINAL_LABELS_PROMOTION_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-final-labels-promotion.v1"
)
TERMINAL_DEFERRAL_FAILURE_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-terminal-deferral-failure.v1"
)
TERMINAL_VALIDATION_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-terminal-validation.v1"
)
SUPPORTED_ASSET_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
CODEX_EXECUTION_HOST = "host"
CODEX_EXECUTION_CONTAINER = "container"
SUPPORTED_CODEX_EXECUTION_MODES = frozenset(
    {CODEX_EXECUTION_HOST, CODEX_EXECUTION_CONTAINER}
)
SUPPORTED_WORKFLOW_SKILLS = frozenset(
    {
        "content-workflow-mesh-segmentation",
    }
)
DEFAULT_CODEX_CONTAINER_IMAGE = "content-workflow-cli:local"
# Even the 36-face two-part acceptance fixture can spend close to an hour in
# evidence and falsification. Keep one turn large enough to finish that work;
# the case budget still bounds continuation turns.
DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS = 7200.0
DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS = 21600.0
# File ctime and Python's wall clock can differ slightly across filesystems or
# CPUs. Memory timing is diagnostic, so tolerate one timestamp-resolution tick.
MEMORY_PROMOTION_GRACE_SECONDS = 1.0
# Target names become top-level directories in the run. Keep them disjoint from
# every launcher/skill-owned top-level artifact and from the residual segment.
RESERVED_TARGET_SEMANTIC_PART_NAMES = frozenset(
    {
        ".agents",
        ".memory",
        ".runtime",
        "agent_prompt.md",
        "child-final.md",
        "child-output.log",
        "continuation_seed",
        "edits",
        "final",
        "fragments",
        "hypotheses",
        "initialization",
        "initialization_gate.json",
        "inputs",
        "live_sequential_gate.json",
        "live_target_gate.json",
        "other",
        "part_lock_manifest.json",
        "part_plan.json",
        "part_review_manifest.json",
        "part_work_queue.json",
        "prepare",
        "raw",
        "request.json",
        "scripts",
        "segments.json",
        "state",
        "target_review_manifest.json",
        "terminal_validation.json",
        "trace",
        "validation",
        "usd-cli.log",
    }
)


def _semantic_part_name_error(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return "must be a non-empty string"
    if Path(value).name != value or value in {".", ".."}:
        return "must be a plain semantic name, not a path"
    if value.casefold() in RESERVED_TARGET_SEMANTIC_PART_NAMES:
        return "conflicts with a reserved run artifact name"
    return None


RECOGNITION_SCOPE_ORDER = {
    "tiny": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "body": 4,
}
LIVE_SEQUENTIAL_GATE_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-live-sequential-gate.v1"
)
# Consecutive continuation turns with no measurable change before giving up.
# Must stay well above one: agents legitimately spend several turns on a single
# hard part before recording anything.
_MAX_IDLE_CONTINUATION_TURNS = 6
INITIALIZER_RUNTIME_GATE_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-initializer-runtime-gate.v1"
)
INITIALIZER_DECISION_SCHEMA_VERSION = "mesh-segmentation-initializer-decision.v1"
INITIALIZER_DECISION_VALIDATION_SCHEMA_VERSION = (
    "mesh-segmentation-initializer-decision-validation.v1"
)
FRAGMENT_UNION_SCHEMA_VERSION = "mesh-segmentation-deterministic-fragment-union.v1"
FRAGMENT_UNION_VALIDATION_SCHEMA_VERSION = (
    "mesh-segmentation-deterministic-fragment-union-validation.v1"
)
FRAGMENT_UNION_METHOD = "eight_view_two_pixel_eroded_closest_visible_fragment_set_union"
INITIALIZER_VIEW_IDS = frozenset(
    {
        "plus_xminus_yplus_z",
        "minus_xminus_yplus_z",
        "plus_xplus_yplus_z",
        "minus_xplus_yplus_z",
        "plus_xminus_yminus_z",
        "minus_xminus_yminus_z",
        "plus_xplus_yminus_z",
        "minus_xplus_yminus_z",
    }
)
SEMANTIC_OVERLAY_SCHEMA_VERSION = "content-agents.mesh-semantic-overlays.v1"
SEMANTIC_OVERLAY_REGISTRATION_SCHEMA_VERSION = (
    "content-agents.mesh-semantic-overlay-registration.v1"
)
MASK_FACE_UNION_SCHEMA_VERSION = "content-agents.mesh-mask-face-union.v1"
MASK_FACE_HIERARCHICAL_VOTE_SCHEMA_VERSION = (
    "content-agents.mesh-mask-hierarchical-vote.v1"
)
DETERMINISTIC_INITIALIZATION_SCHEMA_VERSION = (
    "content-agents.mesh-mask-face-union-validation.v1"
)
HIERARCHICAL_VOTE_VALIDATION_SCHEMA_VERSION = (
    "content-agents.mesh-mask-hierarchical-vote-validation.v1"
)
DETERMINISTIC_INITIALIZATION_ALGORITHM = "registered_mask_nearest_visible_face_union_v1"
HIERARCHICAL_VOTE_INITIALIZATION_ALGORITHM = "semantic_mask_hierarchical_face_vote_v1"
# The initialization validator below accepts the legacy direct-union output and
# the current hierarchical positive/negative vote output for replaying older runs.
COARSE_SAMPLE_SCHEMA_VERSION = "content-agents.mesh-coarse-samples.v1"
SURFACE_SEED_DECISION_SCHEMA_VERSION = "content-agents.mesh-surface-seed-decision.v1"
SURFACE_HIERARCHY_SCHEMA_VERSION = "content-agents.mesh-surface-hierarchy.v1"
DETERMINISTIC_INITIALIZATION_ROLES = (
    "plus_x",
    "minus_x",
    "plus_xminus_yplus_z",
    "minus_xminus_yplus_z",
    "plus_xminus_yminus_z",
    "minus_xminus_yminus_z",
)
MINIMUM_SEMANTIC_OVERLAY_ACCEPTED_VIEWS = 4
MINIMUM_SEMANTIC_FACE_VIEW_SUPPORT = 2
MINIMUM_SEMANTIC_VIEW_SEPARATION_DEGREES = 30.0
SEMANTIC_OVERLAY_DECISION_METHOD = (
    "registered_semantic_overlay_reprojected_proposals_v3"
)
SEMANTIC_OVERLAY_CORROBORATION_UNIT = "same_sampled_3d_surface_point"
SEMANTIC_OVERLAY_VISIBILITY_TEST = "linear_depth_reprojection"
SEMANTIC_OVERLAY_DEPTH_RELATIVE_TOLERANCE = 0.003
SEMANTIC_OVERLAY_DEPTH_NEIGHBORHOOD_PIXELS = 1


@dataclass(frozen=True)
class MeshSegmentationConfig:
    child_launch_profile: ClassVar[str] = "mesh-segmentation.run"
    repo_root: Path
    asset_path: Path
    reference_images: list[Path]
    expected_asset_sha256: str | None = None
    expected_reference_sha256: list[str] = field(default_factory=list)
    workflow_skill: str = "content-workflow-mesh-segmentation"
    target_prim_path: str | None = None
    target_semantic_parts: list[str] = field(default_factory=list)
    continue_from_run: Path | None = None
    asset_root: Path | None = None
    output_dir: Path | None = None
    run_id: str | None = None
    runner: str = RUNNER_CODEX
    model: str | None = None
    model_reasoning_effort: str | None = None
    codex_base_url: str | None = None
    codex_responses_url: str | None = None
    codex_api_key_env: str | None = None
    allow_codex_configured_auth: bool = False
    codex_sandbox_mode: str = CODEX_SANDBOX_WORKSPACE_WRITE
    codex_config: dict[str, object] | None = None
    codex_execution_mode: str = CODEX_EXECUTION_HOST
    allow_unsafe_host_child: bool = False
    codex_container_image: str = DEFAULT_CODEX_CONTAINER_IMAGE
    claude_config: dict[str, object] | None = None
    claude_permission_mode: str = "default"
    claude_max_turns: int | None = None
    claude_execution_mode: str = CLAUDE_EXECUTION_SDK
    scene_tool_timeout_seconds: float = 60.0
    usd_cli_session_id: str | None = None
    child_timeout_seconds: float = DEFAULT_MESH_SEGMENTATION_CHILD_TIMEOUT_SECONDS
    iteration_budget: int = 12
    # Total wall clock across every turn of one case. `child_timeout_seconds`
    # bounds a single turn, so without this the two multiply. 0 disables.
    case_timeout_seconds: float = DEFAULT_MESH_SEGMENTATION_CASE_TIMEOUT_SECONDS
    memory_enabled: bool = True
    memory_root: Path | None = None
    image_gen_backend: str | None = None
    image_gen_model: str | None = None
    image_gen_base_url: str | None = None
    image_gen_api_key_env: str | None = None
    additional_instructions: str | None = None
    dry_run: bool = False
    agent_cwd: Path | None = None
    agent_workspace: Path | None = None
    reference_files: list[Path] = field(default_factory=list)
    parent_usd_cli_session_identity: Path | None = None
    parent_usd_cli_session_identity_sha256: str | None = None

    @property
    def usd_path(self) -> Path:
        """Compatibility field for the shared child-agent runtime protocol."""

        return self.asset_path


@dataclass(frozen=True)
class MeshSegmentationResult:
    run_dir: Path
    request_path: Path
    prompt_path: Path
    child_output_path: Path
    child_final_path: Path
    terminal_validation_path: Path | None
    returncode: int
    completed: bool


def _write_live_sequential_gate(
    gate_path: Path,
    *,
    status: str,
    approved_parts: list[str],
    authorized_segment_ids: list[int],
    next_part: str | None,
    problems: list[str] | None = None,
) -> None:
    _write_json(
        gate_path,
        {
            "schema_version": LIVE_SEQUENTIAL_GATE_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "status": status,
            "approved_count": len(approved_parts),
            "approved_parts": approved_parts,
            "authorized_segment_ids": authorized_segment_ids,
            "next_part": next_part,
            "problems": problems or [],
            "instruction": (
                "Do not introduce a new nonzero segment ID until approved_count "
                "includes the current part."
            ),
        },
        mode=0o644,
    )


def _safe_run_artifact_path(run_dir: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    run_root = run_dir.resolve()
    resolved = (run_root / candidate).resolve()
    if not resolved.is_relative_to(run_root):
        return None
    return resolved


def _manifest_run_artifact_path(run_dir: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    run_root = _lexical_absolute_path(run_dir)
    raw_path = Path(value)
    candidate = _lexical_absolute_path(
        raw_path if raw_path.is_absolute() else run_dir / raw_path
    )
    if not candidate.is_relative_to(run_root):
        return None
    return candidate


def _sorted_existing_paths_by_mtime(paths: list[Path]) -> list[Path]:
    """Sort a filesystem snapshot while tolerating concurrent child cleanup."""

    entries: list[tuple[int, str, Path]] = []
    for path in paths:
        try:
            entries.append((path.stat().st_mtime_ns, str(path), path))
        except OSError:
            continue
    entries.sort()
    return [path for _, _, path in entries]


def _first_nonzero_hypothesis_path(run_dir: Path) -> Path | None:
    paths = [
        *(run_dir / "hypotheses").glob("**/face_labels.u32le"),
        *(run_dir / "edits").glob("**/face_labels.u32le"),
    ]
    if not paths and (run_dir / "final" / "face_labels.u32le").is_file():
        paths.append(run_dir / "final" / "face_labels.u32le")
    for path in _sorted_existing_paths_by_mtime(paths):
        labels, error = _load_u32_labels(path)
        if error is not None or labels is None:
            continue
        if any(value != 0 for value in labels):
            return path
    return None


def _multiview_semantic_role_errors(accepted_roles: list[str]) -> list[str]:
    """Require useful side and elevation diversity, not merely a view count."""

    errors: list[str] = []
    role_set = set(accepted_roles)
    if len(role_set) < MINIMUM_SEMANTIC_OVERLAY_ACCEPTED_VIEWS:
        errors.append("Fewer than four semantic overlays passed registration")
    if not any(role.startswith("plus_x") for role in role_set) or not any(
        role.startswith("minus_x") for role in role_set
    ):
        errors.append("Accepted semantic overlays do not cover both asset sides")
    if not any("plus_z" in role for role in role_set) or not any(
        "minus_z" in role for role in role_set
    ):
        errors.append(
            "Accepted semantic overlays do not cover both upper and lower obliques"
        )
    return errors


def _semantic_seed_policy_errors(decision: dict[str, object]) -> list[str]:
    """Validate that image-generated masks remain corroborated proposals."""

    errors: list[str] = []
    if decision.get("decision_method") != SEMANTIC_OVERLAY_DECISION_METHOD:
        errors.append("Surface seed decision did not use corroborated proposals")
    if decision.get("semantic_mask_trust") != "fallible_proposal_only":
        errors.append("Surface seed decision treats the semantic mask as trusted")
    if decision.get("seed_lock_policy") != "never_lock_from_overlay":
        errors.append("Surface seed decision may incorrectly hard-lock overlay seeds")
    parameters = decision.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("Surface seed decision lacks corroboration parameters")
        return errors
    if (
        parameters.get("minimum_positive_face_view_support")
        != MINIMUM_SEMANTIC_FACE_VIEW_SUPPORT
    ):
        errors.append("Positive semantic face seeds lack two-view corroboration")
    if (
        parameters.get("minimum_negative_face_view_support")
        != MINIMUM_SEMANTIC_FACE_VIEW_SUPPORT
    ):
        errors.append("Negative semantic face seeds lack two-view corroboration")
    if (
        parameters.get("minimum_view_separation_degrees")
        != MINIMUM_SEMANTIC_VIEW_SEPARATION_DEGREES
    ):
        errors.append("Semantic face seeds lack independent camera separation")
    if parameters.get("corroboration_unit") != SEMANTIC_OVERLAY_CORROBORATION_UNIT:
        errors.append("Semantic face seeds do not corroborate one 3D surface point")
    if parameters.get("visibility_test") != SEMANTIC_OVERLAY_VISIBILITY_TEST:
        errors.append("Semantic face seeds do not use depth-tested reprojection")
    if (
        parameters.get("depth_relative_tolerance")
        != SEMANTIC_OVERLAY_DEPTH_RELATIVE_TOLERANCE
    ):
        errors.append("Semantic face seed depth tolerance is not fixed")
    if (
        parameters.get("depth_neighborhood_pixels")
        != SEMANTIC_OVERLAY_DEPTH_NEIGHBORHOOD_PIXELS
    ):
        errors.append("Semantic face seed depth neighborhood is not fixed")
    face_decisions = decision.get("face_decisions")
    if not isinstance(face_decisions, list):
        errors.append("Surface seed decision lacks per-face vote provenance")
        return errors
    for record in face_decisions:
        if not isinstance(record, dict):
            errors.append("Surface seed decision contains an invalid face vote")
            continue
        label = record.get("decision")
        support_key = (
            "positive_view_support" if label == "positive" else "negative_view_support"
        )
        separation_key = (
            "positive_view_separation_degrees"
            if label == "positive"
            else "negative_view_separation_degrees"
        )
        if label not in {"positive", "negative"}:
            continue
        support = record.get(support_key)
        separation = record.get(separation_key)
        if (
            not isinstance(support, int)
            or support < MINIMUM_SEMANTIC_FACE_VIEW_SUPPORT
            or not isinstance(separation, int | float)
            or float(separation) < MINIMUM_SEMANTIC_VIEW_SEPARATION_DEGREES
        ):
            errors.append(f"A {label} semantic face seed is single-view evidence")
    return errors


def _initializer_artifact(
    run_dir: Path,
    payload: dict[str, object],
    *,
    path_key: str,
    digest_key: str,
    label: str,
    errors: list[str],
    sealed_digests: dict[str, str],
) -> Path | None:
    path = _manifest_run_artifact_path(run_dir, payload.get(path_key))
    if path is None:
        errors.append(f"{label} has an unsafe or missing {path_key}")
        return None
    run_root = run_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(run_root) or path.is_symlink():
        errors.append(f"{label} {path_key} must be a regular in-run artifact")
        return None
    if not resolved.is_file():
        errors.append(f"{label} {path_key} is missing: {resolved}")
        return None
    digest = _sha256_file(resolved)
    if payload.get(digest_key) != digest:
        errors.append(f"{label} {digest_key} does not match {path_key}")
    sealed_digests[resolved.relative_to(run_root).as_posix()] = digest
    return resolved


def _load_initializer_ids(
    path: Path,
    *,
    label: str,
    errors: list[str],
) -> np.ndarray | None:
    try:
        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False)
        else:
            values = np.fromfile(path, dtype="<u4")
    except (OSError, ValueError) as exc:
        errors.append(f"Could not read {label}: {exc}")
        return None
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        errors.append(f"{label} must be a one-dimensional integer array")
        return None
    result = values.astype(np.int64, copy=False)
    if not np.array_equal(result, np.unique(result)):
        errors.append(f"{label} must contain sorted unique IDs")
    return result


def _load_initializer_dense_ids(
    path: Path,
    *,
    label: str,
    errors: list[str],
) -> np.ndarray | None:
    try:
        if path.suffix == ".npy":
            values = np.load(path, allow_pickle=False)
        else:
            values = np.fromfile(path, dtype="<u4")
    except (OSError, ValueError) as exc:
        errors.append(f"Could not read {label}: {exc}")
        return None
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        errors.append(f"{label} must be a one-dimensional integer array")
        return None
    if len(values) == 0:
        errors.append(f"{label} must not be empty")
        return None
    return values.astype(np.int64, copy=False)


def _erode_binary_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    eroded = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        eroded = np.logical_and.reduce(
            [
                padded[row : row + eroded.shape[0], column : column + eroded.shape[1]]
                for row in range(3)
                for column in range(3)
            ]
        )
    return eroded


def validate_part_initialization(
    run_dir: Path,
    part_name: str,
    *,
    part_dir: Path | None = None,
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    """Replay one canonical initializer and bind it to the part's rev-000."""

    errors: list[str] = []
    sealed_digests: dict[str, str] = {}
    metadata: dict[str, object] = {"semantic_part": part_name}
    if not part_name or Path(part_name).name != part_name or part_name in {".", ".."}:
        return [f"Unsafe initializer part name: {part_name!r}"], {}, metadata

    run_root = run_dir.resolve()
    if part_dir is None:
        part_dir = _resolve_part_dir(run_root, part_name) or (run_root / part_name)
    part_dir = part_dir.resolve()
    if not part_dir.is_relative_to(run_root):
        return (
            [f"Initializer part directory escapes the run: {part_name!r}"],
            {},
            {"semantic_part": part_name},
        )
    part_label = str(part_dir.relative_to(run_root))
    decision_path = part_dir / "initializer_decision.json"
    decision, decision_error = _load_json_object(
        decision_path,
        label=f"{part_label}/initializer_decision.json",
    )
    if decision_error:
        return [decision_error], {}, metadata
    assert decision is not None
    decision_digest = _sha256_file(decision_path)
    sealed_digests[decision_path.relative_to(run_root).as_posix()] = decision_digest
    route = decision.get("decision")
    segment_id = decision.get("segment_id")
    metadata.update(
        {
            "decision": route,
            "decision_sha256": decision_digest,
            "segment_id": segment_id,
        }
    )
    if not _schema_matches(
        decision.get("schema_version"), INITIALIZER_DECISION_SCHEMA_VERSION
    ):
        errors.append("Initializer decision has an unsupported schema")
    if decision.get("status") != "accepted":
        errors.append("Initializer decision status must be 'accepted'")
    recorded_part = decision.get("semantic_part")
    if not isinstance(recorded_part, str) or _normalize_part_key(
        recorded_part
    ) != _normalize_part_key(part_dir.name):
        errors.append("Initializer decision semantic_part does not match its directory")
    else:
        # The decision names the part; the directory only has to fold to it.
        metadata["semantic_part"] = recorded_part
    if route not in {"image_mask_seed", "direct_agentic_selection"}:
        errors.append("Initializer decision has an unsupported route")
    if (
        not isinstance(segment_id, int)
        or isinstance(segment_id, bool)
        or segment_id <= 0
    ):
        errors.append("Initializer decision segment_id must be a positive integer")

    decision_validation_path = part_dir / "initializer_decision_validation.json"
    decision_validation, validation_error = _load_json_object(
        decision_validation_path,
        label=f"{part_label}/initializer_decision_validation.json",
    )
    if validation_error:
        errors.append(validation_error)
        decision_validation = None
    else:
        assert decision_validation is not None
        sealed_digests[decision_validation_path.relative_to(run_root).as_posix()] = (
            _sha256_file(decision_validation_path)
        )
        # Match the surrounding decision and union manifests, which use
        # `_schema_matches` precisely so a pre-rename run stays readable. An
        # exact comparison here failed those runs on the nested document alone.
        if not _schema_matches(
            decision_validation.get("schema_version"),
            INITIALIZER_DECISION_VALIDATION_SCHEMA_VERSION,
        ):
            errors.append("Initializer decision validation has an unsupported schema")
        if decision_validation.get("status") != "passed":
            errors.append("Initializer decision validation did not pass")
        # The validation must name the part its decision names, exactly. Only
        # the directory spelling is allowed to differ from the semantic name.
        if decision_validation.get("semantic_part") != recorded_part:
            errors.append("Initializer decision validation names another part")
        if decision_validation.get("decision") != route:
            errors.append("Initializer decision validation names another route")
        if decision_validation.get("segment_id") != segment_id:
            errors.append("Initializer decision validation segment_id differs")
        if decision_validation.get("decision_sha256") != decision_digest:
            errors.append("Initializer decision validation has a stale decision digest")
        artifacts = decision_validation.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append("Initializer decision validation has no sealed evidence")
        else:
            for index, artifact in enumerate(artifacts):
                if not isinstance(artifact, dict):
                    errors.append(
                        f"Initializer decision evidence record {index} is invalid"
                    )
                    continue
                _initializer_artifact(
                    run_root,
                    artifact,
                    path_key="path",
                    digest_key="sha256",
                    label=f"initializer decision evidence record {index}",
                    errors=errors,
                    sealed_digests=sealed_digests,
                )

    revision_dir = part_dir / "rev-000"
    revision_labels_path = revision_dir / "face_labels.u32le"
    revision_manifest_path = revision_dir / "edit_manifest.json"
    revision_manifest, revision_error = _load_json_object(
        revision_manifest_path,
        label=f"{part_label}/rev-000/edit_manifest.json",
    )
    if revision_error:
        errors.append(revision_error)
        revision_manifest = None
    if not revision_labels_path.is_file():
        errors.append(f"{part_label}/rev-000/face_labels.u32le is missing")
        revision_digest = None
    else:
        revision_digest = _sha256_file(revision_labels_path)
        sealed_digests[revision_labels_path.relative_to(run_root).as_posix()] = (
            revision_digest
        )
        metadata["rev_000_face_labels_sha256"] = revision_digest
    if revision_manifest is not None:
        sealed_digests[revision_manifest_path.relative_to(run_root).as_posix()] = (
            _sha256_file(revision_manifest_path)
        )
        if revision_manifest.get("status") != "passed":
            errors.append("rev-000 edit manifest did not pass")
        if revision_manifest.get("active_segment_id") != segment_id:
            errors.append("rev-000 active_segment_id differs from the decision")
        if revision_manifest.get("semantic_decision_unit") != "immutable_fragment":
            errors.append("rev-000 did not preserve fragment atomicity")
        if (
            revision_digest is not None
            and revision_manifest.get("face_labels_sha256") != revision_digest
        ):
            errors.append("rev-000 edit manifest has a stale face-label digest")
        for path_key, digest_key in (
            ("parent_labels", "parent_labels_sha256"),
            ("fragment_labels", "fragment_labels_sha256"),
            ("edits", "edits_sha256"),
        ):
            _initializer_artifact(
                run_root,
                revision_manifest,
                path_key=path_key,
                digest_key=digest_key,
                label="rev-000 edit manifest",
                errors=errors,
                sealed_digests=sealed_digests,
            )

    if route != "image_mask_seed":
        return errors, sealed_digests, metadata

    union_dir = part_dir / "initializer-seed" / "union"
    union_manifest_path = union_dir / "manifest.json"
    union_manifest, union_error = _load_json_object(
        union_manifest_path,
        label=f"{part_label}/initializer-seed/union/manifest.json",
    )
    if union_error:
        errors.append(union_error)
        return errors, sealed_digests, metadata
    assert union_manifest is not None
    sealed_digests[union_manifest_path.relative_to(run_root).as_posix()] = _sha256_file(
        union_manifest_path
    )
    if not _schema_matches(
        union_manifest.get("schema_version"), FRAGMENT_UNION_SCHEMA_VERSION
    ):
        errors.append("Deterministic fragment union has an unsupported schema")
    if union_manifest.get("method") != FRAGMENT_UNION_METHOD:
        errors.append("Deterministic fragment union has an unsupported method")
    if union_manifest.get("erosion_pixels") != 2:
        errors.append("Deterministic fragment union must use exactly 2 px erosion")
    if union_manifest.get("cross_view_reduction") != "exact_set_union":
        errors.append("Deterministic fragment union must use exact set union")
    for flag_name in (
        "negative_pixels_used",
        "pixel_ratios_used",
        "cross_view_voting_used",
        "cross_view_negative_veto_used",
    ):
        if union_manifest.get(flag_name) is not False:
            errors.append(f"Deterministic fragment union requires {flag_name}=false")
    if union_manifest.get("active_segment_id") != segment_id:
        errors.append("Deterministic fragment union segment_id differs")

    validation_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="validation",
        digest_key="validation_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    union_validation: dict[str, object] | None = None
    if validation_path is not None:
        union_validation, union_validation_error = _load_json_object(
            validation_path,
            label="deterministic fragment union validation",
        )
        if union_validation_error:
            errors.append(union_validation_error)
    if union_validation is not None:
        if not _schema_matches(
            union_validation.get("schema_version"),
            FRAGMENT_UNION_VALIDATION_SCHEMA_VERSION,
        ):
            errors.append("Fragment-union validation has an unsupported schema")
        if union_validation.get("status") != "passed":
            errors.append("Fragment-union validation did not pass")
        for flag_name in (
            "exactly_expected_views_verified",
            "closest_visible_buffers_verified",
            "per_view_projection_independence_verified",
            "two_pixel_erosion_verified",
            "set_union_replay_verified",
        ):
            if union_validation.get(flag_name) is not True:
                errors.append(f"Fragment-union validation requires {flag_name}=true")
        for flag_name in (
            "negative_pixels_used",
            "pixel_ratios_used",
            "cross_view_voting_used",
            "cross_view_negative_veto_used",
            "agent_confirmation_before_rev_000",
        ):
            if union_validation.get(flag_name) is not False:
                errors.append(f"Fragment-union validation requires {flag_name}=false")

    view_order = union_manifest.get("view_order")
    views = union_manifest.get("views")
    if (
        union_manifest.get("view_count") != 8
        or not isinstance(view_order, list)
        or len(view_order) != 8
        or set(view_order) != INITIALIZER_VIEW_IDS
    ):
        errors.append("Fragment union must contain the eight canonical views")
        view_order = []
    if not isinstance(views, list) or len(views) != 8:
        errors.append("Fragment union must contain eight independent view records")
        views = []
    id_manifest_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="id_buffer_manifest",
        digest_key="id_buffer_manifest_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    id_views: dict[str, dict[str, object]] = {}
    if id_manifest_path is not None:
        id_manifest, id_manifest_error = _load_json_object(
            id_manifest_path,
            label="closest-visible ID-buffer manifest",
        )
        if id_manifest_error:
            errors.append(id_manifest_error)
        elif id_manifest is not None:
            if id_manifest.get("schema_version") != "mesh-segmentation-id-buffers.v1":
                errors.append("ID-buffer manifest has an unsupported schema")
            id_records = id_manifest.get("views")
            if not isinstance(id_records, list) or len(id_records) != 8:
                errors.append("ID-buffer manifest must contain eight views")
            else:
                for id_record in id_records:
                    if not isinstance(id_record, dict):
                        errors.append("ID-buffer view records must be objects")
                        continue
                    id_name = id_record.get("name")
                    if not isinstance(id_name, str) or id_name in id_views:
                        errors.append("ID-buffer view names must be unique strings")
                        continue
                    if id_record.get("closest_visible_hit_only") is not True:
                        errors.append(
                            f"ID-buffer view {id_name!r} is not closest-visible"
                        )
                    id_views[id_name] = id_record
                if set(id_views) != INITIALIZER_VIEW_IDS:
                    errors.append("ID-buffer manifest lacks the canonical view set")
    view_sets: dict[str, np.ndarray] = {}
    for index, record in enumerate(views):
        if not isinstance(record, dict):
            errors.append(f"Fragment-union view record {index} is invalid")
            continue
        view_id = record.get("view_id")
        if not isinstance(view_id, str) or view_id not in INITIALIZER_VIEW_IDS:
            errors.append(f"Fragment-union view record {index} has an invalid ID")
            continue
        if view_id in view_sets:
            errors.append(f"Fragment-union view {view_id!r} is duplicated")
            continue
        if record.get("erosion_pixels") != 2:
            errors.append(f"Fragment-union view {view_id!r} did not use 2 px erosion")
        npy_path = _initializer_artifact(
            run_root,
            record,
            path_key="selected_fragment_ids_npy",
            digest_key="selected_fragment_ids_npy_sha256",
            label=f"fragment-union view {view_id}",
            errors=errors,
            sealed_digests=sealed_digests,
        )
        raw_path = _initializer_artifact(
            run_root,
            record,
            path_key="selected_fragment_ids_raw",
            digest_key="selected_fragment_ids_raw_sha256",
            label=f"fragment-union view {view_id}",
            errors=errors,
            sealed_digests=sealed_digests,
        )
        mask_path = _initializer_artifact(
            run_root,
            record,
            path_key="registered_mask",
            digest_key="registered_mask_sha256",
            label=f"fragment-union view {view_id}",
            errors=errors,
            sealed_digests=sealed_digests,
        )
        fragment_buffer_path = _initializer_artifact(
            run_root,
            record,
            path_key="fragment_id_buffer",
            digest_key="fragment_id_buffer_sha256",
            label=f"fragment-union view {view_id}",
            errors=errors,
            sealed_digests=sealed_digests,
        )
        eroded_mask_path = _initializer_artifact(
            run_root,
            record,
            path_key="eroded_mask",
            digest_key="eroded_mask_sha256",
            label=f"fragment-union view {view_id}",
            errors=errors,
            sealed_digests=sealed_digests,
        )
        for path_key, digest_key in (
            ("registration_manifest", "registration_manifest_sha256"),
            ("source_render", "source_render_sha256"),
            ("chosen_pixel_overlay", "chosen_pixel_overlay_sha256"),
            ("selected_fragment_projection", "selected_fragment_projection_sha256"),
        ):
            _initializer_artifact(
                run_root,
                record,
                path_key=path_key,
                digest_key=digest_key,
                label=f"fragment-union view {view_id}",
                errors=errors,
                sealed_digests=sealed_digests,
            )
        npy_ids = (
            _load_initializer_ids(
                npy_path,
                label=f"{view_id} selected fragment IDs",
                errors=errors,
            )
            if npy_path is not None
            else None
        )
        raw_ids = (
            _load_initializer_ids(
                raw_path,
                label=f"{view_id} raw selected fragment IDs",
                errors=errors,
            )
            if raw_path is not None
            else None
        )
        if npy_ids is not None and raw_ids is not None:
            if not np.array_equal(npy_ids, raw_ids):
                errors.append(f"Fragment-union view {view_id!r} ID files differ")
            if record.get("selected_fragment_count") != len(npy_ids):
                errors.append(f"Fragment-union view {view_id!r} selected count differs")
            replayed_ids: np.ndarray | None = None
            if mask_path is not None and fragment_buffer_path is not None:
                try:
                    mask = (
                        np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                        >= 128
                    )
                    fragment_buffer = np.load(
                        fragment_buffer_path,
                        allow_pickle=False,
                    )
                    if (
                        fragment_buffer.ndim != 2
                        or not np.issubdtype(fragment_buffer.dtype, np.integer)
                        or fragment_buffer.shape != mask.shape
                    ):
                        errors.append(
                            f"Fragment-union view {view_id!r} mask/ID shapes differ"
                        )
                    else:
                        eroded = _erode_binary_mask(mask, 2)
                        replayed_ids = np.unique(
                            fragment_buffer[eroded & (fragment_buffer >= 0)]
                        ).astype(np.int64)
                        if record.get("eroded_positive_pixel_count") != int(
                            np.count_nonzero(eroded)
                        ):
                            errors.append(
                                f"Fragment-union view {view_id!r} eroded count differs"
                            )
                        if record.get("chosen_visible_pixel_count") != int(
                            np.count_nonzero(eroded & (fragment_buffer >= 0))
                        ):
                            errors.append(
                                f"Fragment-union view {view_id!r} visible count differs"
                            )
                        if eroded_mask_path is not None:
                            saved_eroded = (
                                np.asarray(
                                    Image.open(eroded_mask_path).convert("L"),
                                    dtype=np.uint8,
                                )
                                >= 128
                            )
                            if not np.array_equal(eroded, saved_eroded):
                                errors.append(
                                    f"Fragment-union view {view_id!r} saved erosion "
                                    "does not replay"
                                )
                except (OSError, ValueError) as exc:
                    errors.append(
                        f"Could not replay fragment-union view {view_id!r}: {exc}"
                    )
            id_record = id_views.get(view_id)
            if id_record is not None:
                channels = id_record.get("channels")
                fragment_channel = (
                    channels.get("fragment_ids") if isinstance(channels, dict) else None
                )
                if not isinstance(fragment_channel, dict):
                    errors.append(f"ID-buffer view {view_id!r} lacks fragment_ids")
                else:
                    id_raw_path = _manifest_run_artifact_path(
                        run_root,
                        fragment_channel.get("raw"),
                    )
                    if (
                        fragment_buffer_path is None
                        or id_raw_path is None
                        or id_raw_path.resolve() != fragment_buffer_path.resolve()
                        or fragment_channel.get("raw_sha256")
                        != _sha256_file(fragment_buffer_path)
                    ):
                        errors.append(
                            f"Fragment-union view {view_id!r} is not bound to "
                            "its ID-buffer manifest"
                        )
            if replayed_ids is not None and not np.array_equal(
                replayed_ids,
                npy_ids,
            ):
                errors.append(
                    f"Fragment-union view {view_id!r} selected IDs do not replay "
                    "from the eroded mask"
                )
            view_sets[view_id] = npy_ids
    if view_order and [record.get("view_id") for record in views] != view_order:
        errors.append("Fragment-union view records do not match view_order")

    raw_union = (
        np.unique(np.concatenate(list(view_sets.values())))
        if len(view_sets) == 8
        else None
    )
    raw_union_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="raw_union_fragment_ids",
        digest_key="raw_union_fragment_ids_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    recorded_raw_union = (
        _load_initializer_ids(
            raw_union_path,
            label="raw fragment union",
            errors=errors,
        )
        if raw_union_path is not None
        else None
    )
    if raw_union is not None and recorded_raw_union is not None:
        if not np.array_equal(raw_union, recorded_raw_union):
            errors.append("Saved per-view fragment sets do not replay to raw union")
        if union_manifest.get("raw_union_fragment_count") != len(raw_union):
            errors.append("Raw fragment-union count differs")

    fragment_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="fragment_labels",
        digest_key="fragment_labels_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    parent_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="parent_labels",
        digest_key="parent_labels_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    fragment_labels: np.ndarray | None = None
    parent_labels: np.ndarray | None = None
    if fragment_path is not None:
        fragment_labels = _load_initializer_dense_ids(
            fragment_path,
            label="fragment labels",
            errors=errors,
        )
    if parent_path is not None:
        parent_labels = _load_initializer_dense_ids(
            parent_path,
            label="parent labels",
            errors=errors,
        )
    applied_union: np.ndarray | None = None
    expected_labels: np.ndarray | None = None
    if (
        raw_union is not None
        and fragment_labels is not None
        and parent_labels is not None
    ):
        if len(fragment_labels) != len(parent_labels) or len(fragment_labels) == 0:
            errors.append("Fragment and parent label arrays have incompatible shapes")
        elif len(raw_union) and int(raw_union.max(initial=-1)) > int(
            fragment_labels.max(initial=-1)
        ):
            errors.append("Raw fragment union contains an out-of-range fragment ID")
        else:
            background_id = union_manifest.get("background_segment_id")
            if not isinstance(background_id, int):
                errors.append("Fragment union lacks an integer background segment ID")
            else:
                immutable_faces = (parent_labels != background_id) & (
                    parent_labels != segment_id
                )
                immutable_fragments = np.unique(fragment_labels[immutable_faces])
                applied_union = np.setdiff1d(
                    raw_union,
                    immutable_fragments,
                    assume_unique=True,
                )
                expected_labels = parent_labels.copy()
                expected_labels[np.isin(fragment_labels, applied_union)] = segment_id

    union_npy_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="union_fragment_ids_npy",
        digest_key="union_fragment_ids_npy_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    union_raw_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="union_fragment_ids_raw",
        digest_key="union_fragment_ids_raw_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    recorded_union_npy = (
        _load_initializer_ids(
            union_npy_path,
            label="applied fragment union",
            errors=errors,
        )
        if union_npy_path is not None
        else None
    )
    recorded_union_raw = (
        _load_initializer_ids(
            union_raw_path,
            label="raw applied fragment union",
            errors=errors,
        )
        if union_raw_path is not None
        else None
    )
    if applied_union is not None:
        for label, recorded in (
            ("npy", recorded_union_npy),
            ("raw", recorded_union_raw),
        ):
            if recorded is not None and not np.array_equal(applied_union, recorded):
                errors.append(f"Replayed applied union differs from its {label} file")
        if union_manifest.get("union_fragment_count") != len(applied_union):
            errors.append("Applied fragment-union count differs")

    expected_path = _initializer_artifact(
        run_root,
        union_manifest,
        path_key="expected_face_labels",
        digest_key="expected_face_labels_sha256",
        label="deterministic fragment union",
        errors=errors,
        sealed_digests=sealed_digests,
    )
    if expected_path is not None and expected_labels is not None:
        expected_bytes = expected_labels.astype("<u4").tobytes()
        if expected_path.read_bytes() != expected_bytes:
            errors.append("Expected face labels do not replay from the fragment union")
        if (
            revision_labels_path.is_file()
            and revision_labels_path.read_bytes() != expected_bytes
        ):
            errors.append(
                "rev-000 is not byte-identical to the deterministic fragment union"
            )
    return errors, sealed_digests, metadata


def _gate_sealed_digests(gate_path: Path) -> dict[str, str] | None:
    """The digests a previous launcher pass accepted for this part."""

    if not gate_path.is_file():
        return None
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(gate, dict) or gate.get("status") != "accepted":
        return None
    digests = gate.get("sealed_digests")
    if not isinstance(digests, dict) or not digests:
        return None
    return {str(name): str(value) for name, value in digests.items()}


_GATE_IDENTITY_FIELDS = (
    "semantic_part",
    "decision",
    "decision_sha256",
    "segment_id",
    "rev_000_face_labels_sha256",
)


def _gate_identity(gate_path: Path, part_name: str) -> dict[str, object]:
    """The identity fields terminal validation compares, read back from disk."""

    identity: dict[str, object] = {"semantic_part": part_name}
    if not gate_path.is_file():
        return identity
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return identity
    if not isinstance(gate, dict):
        return identity
    for name in _GATE_IDENTITY_FIELDS:
        if name in gate:
            identity[name] = gate[name]
    identity["semantic_part"] = gate.get("semantic_part", part_name)
    return identity


def _gate_saw_lock(gate_path: Path) -> bool:
    """Whether a previous launcher pass recorded this part as locked."""

    if not gate_path.is_file():
        return False
    try:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(gate, dict) and gate.get("locked_observed") is True


def _write_initializer_runtime_gate(
    path: Path,
    *,
    status: str,
    metadata: dict[str, object],
    problems: list[str] | None = None,
    sealed_digests: dict[str, str] | None = None,
    locked_observed: bool = False,
) -> None:
    _write_json(
        path,
        {
            "schema_version": INITIALIZER_RUNTIME_GATE_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "status": status,
            **metadata,
            "problems": problems or [],
            "sealed_digests": sealed_digests or {},
            # Sticky: once observed, a later pass must not read this as unlocked.
            "locked_observed": locked_observed or _gate_saw_lock(path),
            "instruction": (
                "The launcher independently validates and seals the chosen "
                "initializer route before accepting rev-000."
            ),
        },
        mode=0o644,
    )


_WARNED_LEGACY_SCHEMAS: set[str] = set()
# Kept in step with `mesh_geometry.RENAMED_SCHEMA_STEMS`; the skill scripts and
# the launcher must accept the same set, or an artifact readable by one is
# rejected by the other -- the asymmetry this shim exists to remove.
RENAMED_SCHEMA_STEMS = frozenset(
    {
        "consistency-regions",
        "consistency-sheet",
        "falsification-plan",
        "falsification-review",
        "falsification-validation",
        "initializer-decision",
        "initializer-decision-validation",
        "initializer-runtime-gate",
        "revision-consistency-review",
        "selected-only-stage",
    }
)


def _schema_matches(value: object, expected: str) -> bool:
    """Accept a schema version, tolerating the retired `-v5-` experiment infix.

    The workflow's artifact schemas were briefly published as
    `mesh-segmentation-v5-<name>.vN`, where `v5` named an experiment rather than
    a version (the real version is the `.vN` suffix). New artifacts omit it;
    runs recorded before the rename are still readable.
    """

    if not isinstance(value, str):
        return False
    if value == expected:
        return True
    # Only the schemas that actually carried the infix. Deriving the legacy
    # spelling mechanically also accepted versions no producer ever emitted --
    # `mesh-segmentation-v5-fragment-evidence.v1` and friends -- widening the
    # compat surface this branch promises to remove.
    prefix, separator, remainder = expected.partition("mesh-segmentation-")
    if not separator:
        return False
    if remainder.rsplit(".", 1)[0] not in RENAMED_SCHEMA_STEMS:
        return False
    legacy = f"{prefix}mesh-segmentation-v5-{remainder}"
    if value != legacy:
        return False
    # Accepting two spellings silently is how a compat branch outlives the
    # runs it exists for: nobody learns anything is still producing the old
    # one, and removing the branch later breaks them without warning. Say so
    # once per schema. The skill scripts share this tolerance through
    # `mesh_geometry.schema_matches`, so a pre-rename artifact stays
    # re-validatable by them -- which is what makes removal a scheduled
    # decision rather than a silent break.
    if legacy not in _WARNED_LEGACY_SCHEMAS:
        _WARNED_LEGACY_SCHEMAS.add(legacy)
        print(
            f"content-workflow-cli: accepting retired schema {legacy!r} for "
            f"{expected!r}; this compatibility branch is scheduled for removal",
            file=sys.stderr,
            flush=True,
        )
    return True


def _normalize_part_key(value: str) -> str:
    """Fold a semantic part name or directory name to one comparison key.

    Semantic part names come from the caller's vocabulary and legitimately
    contain spaces and mixed case (`Car Chassis`). Agents materialize them as
    filesystem-safe directories and often prefix a processing order
    (`08_Car_Chassis`, `support_frame`). Comparing folded keys lets the gate
    bind a decision to its directory without dictating a spelling.
    """

    text = re.sub(r"^\d+\s*[._-]+\s*", "", value.strip().lower())
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


# Part evidence is accepted at the run root (`<part>/`) or collected under a
# `parts/` directory. Both layouts occur in practice; the gate must see either,
# because a layout it cannot enumerate is a layout it silently fails to check.
PART_DIRECTORY_ROOTS = ("", "parts")


def _part_directories(run_dir: Path) -> dict[str, Path]:
    """Map each part key to its evidence directory across supported layouts."""

    run_dir = run_dir.resolve()
    directories: dict[str, Path] = {}
    for container_name in PART_DIRECTORY_ROOTS:
        container = run_dir / container_name if container_name else run_dir
        if not container.is_dir():
            continue
        for candidate in sorted(container.iterdir()):
            if not candidate.is_dir():
                continue
            # Deliberate: a directory becomes a part only once it carries a
            # decision or rev-000 labels. Before that there is nothing for the
            # gate to seal or the watchdog to compare against, and scratch or
            # experimental directories would otherwise be gated as parts. A
            # part holding only seed renders or a completion record is
            # therefore invisible here on purpose -- terminal validation is
            # what requires the missing evidence, and it names it.
            if not (
                (candidate / "initializer_decision.json").is_file()
                or (candidate / "rev-000" / "face_labels.u32le").is_file()
            ):
                continue
            resolved = candidate.resolve()
            if resolved != candidate or not resolved.is_relative_to(run_dir):
                # Refuse symlinked or escaping part directories outright.
                continue
            key = _normalize_part_key(candidate.name)
            # A root-level directory wins over a nested duplicate so the
            # historical layout stays authoritative when both are present.
            if key and (key not in directories or not container_name):
                directories[key] = resolved
    return directories


def _resolve_part_dir(run_dir: Path, part_name: str) -> Path | None:
    """Resolve one expected semantic part to its evidence directory."""

    return _part_directories(run_dir).get(_normalize_part_key(part_name))


def _initializer_part_names(run_dir: Path) -> list[str]:
    return sorted(path.name for path in _part_directories(run_dir).values())


def _make_initialization_watchdog(
    run_dir: Path,
    *,
    expected_parts: list[str] | None = None,
) -> Callable[[], WatchdogFailure | None]:
    """Validate and seal each adaptive initializer before refinement."""

    run_root = run_dir.resolve()
    pending_identities: dict[str, tuple[tuple[str, int, int], ...]] = {}
    accepted: dict[str, dict[str, str]] = {}
    reseals: dict[str, int] = {}
    # In targeted mode only the frozen vocabulary is deliverable. Agents also
    # create directories for experiments they reject, and gating those killed
    # runs over work that was correctly discarded.
    gated_keys = {_normalize_part_key(name) for name in expected_parts or []}
    # A part that has ever locked stays locked for this launcher. Reading the
    # completion file's current presence let a child delete it, reseal the
    # evidence as if the part were still provisional, and recreate it --
    # walking straight through the anti-backfill guard.
    ever_locked: set[str] = set()
    # Freeze the exact initializer corpus the launcher observed when a part
    # first became locked. The mutable on-disk runtime gate is an audit record,
    # not a substitute for this in-memory provenance boundary.
    observed_lock_seals: dict[str, dict[str, str]] = {}
    # Seed from the gates on disk so a restart resumes with what the previous
    # pass accepted. Without this the loop below takes the initial-seal path
    # with nothing to compare against -- exactly the state a backfilling child
    # would engineer by deleting its completion record and waiting for one.
    for _key, _part_dir in _part_directories(run_root).items():
        _gate = _part_dir / "initializer_runtime_gate.json"
        if _gate_saw_lock(_gate):
            ever_locked.add(_key)
        _recorded = _gate_sealed_digests(_gate)
        if _recorded is not None:
            accepted[_key] = _recorded
            if _key in ever_locked:
                observed_lock_seals[_key] = dict(_recorded)

    def reject(
        part_dir: Path,
        part_name: str,
        problems: list[str],
        metadata: dict[str, object],
        sealed_digests: dict[str, str],
    ) -> WatchdogFailure | None:
        """Record an integrity violation without destroying the run.

        Killing the child here discarded every other part's completed work,
        including parts that were correct, and bought nothing: terminal
        validation re-runs these exact checks, so a genuinely corrupt run still
        fails. Ending mid-flight only destroyed the evidence needed to diagnose
        it. The gate stays the detector; terminal validation is the enforcer.
        """

        _write_initializer_runtime_gate(
            part_dir / "initializer_runtime_gate.json",
            status="rejected",
            metadata=metadata,
            problems=problems,
            sealed_digests=sealed_digests,
        )
        return None

    def watchdog() -> WatchdogFailure | None:
        for key, part_dir in sorted(_part_directories(run_root).items()):
            if gated_keys and key not in gated_keys:
                continue
            part_name = part_dir.name
            gate_path = part_dir / "initializer_runtime_gate.json"
            if (part_dir / "part_completion.json").is_file():
                ever_locked.add(key)
            elif key not in ever_locked and _gate_saw_lock(gate_path):
                # Seeded from the gate so a launcher restart does not forget.
                # In memory alone, a child could delete the completion record,
                # wait for a restart, and then reseal as though provisional.
                ever_locked.add(key)
            if key in accepted:
                changed = [
                    relative
                    for relative, digest in accepted[key].items()
                    if not (run_root / relative).is_file()
                    or _sha256_file(run_root / relative) != digest
                ]
                if changed:
                    # A part being rebuilt from scratch has no rev-000 yet.
                    # Return it to the waiting state so the normal path seals
                    # the replacement once it lands.
                    if not (
                        (part_dir / "initializer_decision.json").is_file()
                        and (part_dir / "rev-000" / "face_labels.u32le").is_file()
                    ):
                        if key in ever_locked:
                            # Dropping the seal here would let the child
                            # recreate a self-consistent corpus and take the
                            # initial-seal path, which is backfill by another
                            # route.
                            reject(
                                part_dir,
                                part_name,
                                [
                                    "Sealed initializer artifacts disappeared "
                                    "after the part locked: " + ", ".join(changed)
                                ],
                                {"semantic_part": part_name},
                                accepted[key],
                            )
                            return WatchdogFailure(
                                f"Part {part_name} lost sealed initializer "
                                f"artifacts after locking: {', '.join(changed)}",
                                fatal=True,
                            )
                        accepted.pop(key, None)
                        pending_identities.pop(key, None)
                        _write_initializer_runtime_gate(
                            gate_path,
                            status="waiting_for_rev_000",
                            metadata={
                                "semantic_part": part_name,
                                "superseded_artifacts": changed,
                            },
                        )
                        continue
                    # rev-000 is provisional by contract, so re-initializing a
                    # part that has not been locked is legitimate correction,
                    # not tampering. Re-validate and re-seal it, recording the
                    # supersession so the audit trail keeps every version.
                    # Killing the run here discarded otherwise-good work: the
                    # revised initializer still has to pass the same checks.
                    problems, sealed, metadata = validate_part_initialization(
                        run_root,
                        part_name,
                        part_dir=part_dir,
                    )
                    if problems:
                        # Immutability binds once a part is locked. Before
                        # that, rev-000 is explicitly provisional, so an
                        # inconsistent revision mid-correction is the agent's
                        # to fix, not grounds for destroying the whole run.
                        if key not in ever_locked:
                            accepted.pop(key, None)
                            pending_identities.pop(key, None)
                            _write_initializer_runtime_gate(
                                gate_path,
                                status="waiting_for_rev_000",
                                metadata={
                                    "semantic_part": part_name,
                                    "superseded_artifacts": changed,
                                },
                                problems=problems,
                            )
                            continue
                        reject(
                            part_dir,
                            part_name,
                            [
                                "Locked part artifacts changed and no longer "
                                "validate ("
                                + ", ".join(changed)
                                + "): "
                                + "; ".join(problems)
                            ],
                            metadata,
                            accepted[key],
                        )
                        continue
                    # A locked part's evidence is sealed. Rewriting it and
                    # re-validating is backfill: the corpus stays internally
                    # consistent, so terminal validation -- which only re-runs
                    # these same checks -- cannot distinguish it from evidence
                    # observed while the work happened. This gate is the only
                    # detector, so here it has to be fatal. Before the lock,
                    # rev-000 is provisional by contract and a rewrite is
                    # legitimate correction, which is why that path stays
                    # advisory.
                    if key in ever_locked:
                        reject(
                            part_dir,
                            part_name,
                            [
                                "Sealed initializer evidence was rewritten "
                                "after the part locked: " + ", ".join(changed)
                            ],
                            metadata,
                            accepted[key],
                        )
                        return WatchdogFailure(
                            f"Part {part_name} resealed initializer evidence "
                            f"after locking: {', '.join(changed)}",
                            fatal=True,
                        )
                    reseals[key] = reseals.get(key, 0) + 1
                    accepted[key] = sealed
                    pending_identities.pop(key, None)
                    _write_initializer_runtime_gate(
                        gate_path,
                        status="accepted",
                        metadata={
                            **metadata,
                            "reseal_count": reseals[key],
                            "superseded_artifacts": changed,
                        },
                        sealed_digests=sealed,
                    )
                elif key in ever_locked and not _gate_saw_lock(gate_path):
                    # A part seals as soon as its decision and rev-000 agree,
                    # which is long before the falsification step writes
                    # part_completion.json. The lock therefore appears on a
                    # later poll that finds nothing changed and used to fall
                    # straight through, leaving `locked_observed` false on disk
                    # for the rest of the run -- so the restart guard never
                    # fired in any real lifecycle. Persist it the moment the
                    # lock shows up.
                    # Carry the existing identity forward. `_write_initializer_
                    # runtime_gate` rewrites the whole document from metadata,
                    # so passing only the part name dropped decision,
                    # decision_sha256, segment_id and
                    # rev_000_face_labels_sha256 -- the exact four fields
                    # terminal validation compares to decide the gate is stale.
                    # Every locked part then looked stale, took the
                    # watchdog-race path, and had its gate silently rewritten,
                    # so the cross-check never ran.
                    observed_lock_seals.setdefault(key, dict(accepted[key]))
                    _write_initializer_runtime_gate(
                        gate_path,
                        status="accepted",
                        metadata=_gate_identity(gate_path, part_name),
                        sealed_digests=accepted[key],
                        locked_observed=True,
                    )
                continue

            decision_path = part_dir / "initializer_decision.json"
            revision_path = part_dir / "rev-000" / "face_labels.u32le"
            if not decision_path.is_file() or not revision_path.is_file():
                _write_initializer_runtime_gate(
                    gate_path,
                    status="waiting_for_rev_000",
                    metadata={"semantic_part": part_name},
                )
                continue
            identity_paths = (
                decision_path,
                part_dir / "initializer_decision_validation.json",
                revision_path,
                part_dir / "rev-000" / "edit_manifest.json",
            )
            identity = tuple(
                (
                    str(path.relative_to(run_root)),
                    path.stat().st_size if path.is_file() else -1,
                    path.stat().st_mtime_ns if path.is_file() else -1,
                )
                for path in identity_paths
            )
            if pending_identities.get(key) != identity:
                pending_identities[key] = identity
                _write_initializer_runtime_gate(
                    gate_path,
                    status="validating",
                    metadata={"semantic_part": part_name},
                )
                continue
            problems, sealed_digests, metadata = validate_part_initialization(
                run_root,
                part_name,
                part_dir=part_dir,
            )
            if problems:
                # rev-000 is provisional until the part locks, so an
                # initializer that does not yet validate is the agent's to
                # repair. Refuse to seal it and say why, but do not destroy the
                # run's other parts over one unfinished revision. Terminal
                # validation still re-runs these checks, so nothing invalid can
                # pass by simply never being sealed.
                if key not in ever_locked:
                    _write_initializer_runtime_gate(
                        gate_path,
                        status="waiting_for_rev_000",
                        metadata={"semantic_part": part_name},
                        problems=problems,
                    )
                    continue
                reject(part_dir, part_name, problems, metadata, sealed_digests)
                continue
            accepted[key] = sealed_digests
            if key in ever_locked:
                observed_lock_seals.setdefault(key, dict(sealed_digests))
            _write_initializer_runtime_gate(
                gate_path,
                status="accepted",
                metadata=metadata,
                sealed_digests=sealed_digests,
                locked_observed=key in ever_locked,
            )
        return None

    def launcher_observed_lock_seals() -> Mapping[str, Mapping[str, str]]:
        """Expose immutable exact seals for locks observed by this watchdog."""

        return MappingProxyType(
            {
                key: MappingProxyType(dict(seals))
                for key, seals in observed_lock_seals.items()
            }
        )

    setattr(
        watchdog,
        "launcher_observed_lock_seals",
        launcher_observed_lock_seals,
    )
    return watchdog


def _watchdog_observed_lock_seals(
    watchdog: Callable[[], WatchdogFailure | None] | None,
) -> dict[str, dict[str, str]]:
    """Read the in-memory lock provenance exposed by the live watchdog."""

    getter = getattr(watchdog, "launcher_observed_lock_seals", None)
    if not callable(getter):
        return {}
    observed = getter()
    if not isinstance(observed, Mapping):
        return {}
    snapshot: dict[str, dict[str, str]] = {}
    for key, seals in observed.items():
        if not isinstance(key, str) or not isinstance(seals, Mapping):
            return {}
        copied = dict(seals)
        if not copied or any(
            not isinstance(path, str) or not isinstance(digest, str)
            for path, digest in copied.items()
        ):
            return {}
        snapshot[key] = copied
    return snapshot


def _validate_initializer_runtime_gates(
    run_dir: Path,
    *,
    expected_parts: list[str],
) -> list[str]:
    errors: list[str] = []
    run_root = run_dir.resolve()
    directories = _part_directories(run_root)
    # Key every part by its folded name so an expected part and the directory
    # that materializes it are validated once, not twice. Expected parts are
    # kept even without a directory so a part that was never started still
    # fails here rather than passing unnoticed.
    targets: dict[str, str] = {}
    for part_name in expected_parts:
        targets.setdefault(_normalize_part_key(part_name), part_name)
    if not targets:
        # Recognition mode has no frozen vocabulary, so every discovered part
        # directory is a deliverable.
        for key, part_dir in directories.items():
            targets.setdefault(key, part_dir.name)
    # In targeted mode the request's vocabulary is authoritative. Agents also
    # create working directories for rejected experiments, and those carry an
    # initializer_decision.json; validating them as deliverables failed runs
    # over scratch work that was correctly discarded.
    for key, part_name in sorted(targets.items(), key=lambda item: item[1]):
        if (
            not part_name
            or Path(part_name).name != part_name
            or part_name in {".", ".."}
        ):
            errors.append(f"Unsafe initializer part name: {part_name!r}")
            continue
        part_dir = directories.get(key, run_root / part_name)
        problems, sealed_digests, metadata = validate_part_initialization(
            run_dir,
            part_name,
            part_dir=part_dir,
        )
        errors.extend(
            f"{part_name} initializer runtime validation: {problem}"
            for problem in problems
        )
        gate_path = part_dir / "initializer_runtime_gate.json"
        gate, gate_error = _load_json_object(
            gate_path,
            label=f"{part_dir.name}/initializer_runtime_gate.json",
        )
        if gate_error:
            errors.append(gate_error)
            continue
        assert gate is not None
        stale = (
            not _schema_matches(
                gate.get("schema_version"), INITIALIZER_RUNTIME_GATE_SCHEMA_VERSION
            )
            or gate.get("status") != "accepted"
            or gate.get("sealed_digests") != sealed_digests
            or any(
                gate.get(field) != metadata.get(field)
                for field in (
                    "semantic_part",
                    "decision",
                    "decision_sha256",
                    "segment_id",
                    "rev_000_face_labels_sha256",
                )
            )
        )
        if not stale:
            continue
        if problems:
            # The gate is stale and the part does not validate now either, so
            # report the gate state alongside the validation errors above.
            errors.append(
                f"{part_name} initializer runtime gate does not match its "
                "initializer and the initializer is invalid"
            )
            continue
        if gate.get("status") == "rejected":
            # A rejection is a finding, not a stale write. Internal consistency
            # at the end is the *premise* of backfill, so re-validating cleanly
            # here says nothing about what the live gate observed -- and
            # overwriting would erase the only record of it. Keep the verdict
            # and report it.
            recorded = gate.get("problems")
            detail = "; ".join(str(item) for item in recorded) if recorded else ""
            errors.append(
                f"{part_name} initializer runtime gate recorded a rejection"
                + (f": {detail}" if detail else "")
            )
            continue
        # The watchdog polls, so a part finalized just before the child exits
        # can leave a stale gate even though nothing is wrong. Terminal
        # validation has just re-run the identical checks and found no problem,
        # so finalize the gate here rather than failing the run on a race. A
        # missing gate file is still an error: that means it was never gated.
        _write_initializer_runtime_gate(
            gate_path,
            status="accepted",
            metadata={**metadata, "finalized_at_terminal_validation": True},
            sealed_digests=sealed_digests,
        )
    return errors


def _validate_part_semantic_overlay_evidence(
    run_dir: Path,
    *,
    part: dict[str, object],
    lock: dict[str, object],
) -> list[str]:
    """Require distrustful canonical-plus-adaptive overlay evidence per part."""

    if part.get("role") == "body_fallback":
        return []
    name = str(part.get("name", part.get("part_name")))
    evidence = lock.get("semantic_overlay_evidence")
    if not isinstance(evidence, dict):
        return [f"Target {name} lock lacks semantic_overlay_evidence"]

    loaded: dict[str, tuple[Path, dict[str, object]]] = {}
    errors: list[str] = []
    for key, label in (
        ("overlay_manifest", "semantic overlay manifest"),
        ("registration_manifest", "semantic overlay registration"),
        ("projection_manifest", "semantic mask-to-face projection"),
    ):
        path = _safe_run_artifact_path(run_dir, evidence.get(key))
        if path is None or not path.is_file():
            errors.append(f"Target {name} {label} is missing or unsafe")
            continue
        payload, payload_error = _load_json_object(path, label=f"{name} {label}")
        if payload_error:
            errors.append(payload_error)
            continue
        assert payload is not None
        loaded[key] = (path, payload)
    if errors:
        return errors

    overlay_path, overlays = loaded["overlay_manifest"]
    registration_path, registration = loaded["registration_manifest"]
    _projection_path, projection = loaded["projection_manifest"]
    if overlays.get("schema_version") != SEMANTIC_OVERLAY_SCHEMA_VERSION:
        errors.append(f"Target {name} overlay manifest has an unsupported schema")
    if overlays.get("target_semantic_part") != name:
        errors.append(f"Target {name} overlay manifest names a different part")
    overlay_records = overlays.get("records")
    if not isinstance(overlay_records, list):
        errors.append(f"Target {name} overlay records must be an array")
        valid_overlay_records: list[dict[str, object]] = []
    else:
        valid_overlay_records = [
            record for record in overlay_records if isinstance(record, dict)
        ]
        if len(valid_overlay_records) != len(overlay_records):
            errors.append(f"Target {name} overlay records must all be objects")
    overlay_roles = [str(record.get("role")) for record in valid_overlay_records]
    canonical_roles = list(DETERMINISTIC_INITIALIZATION_ROLES)
    if overlay_roles[: len(canonical_roles)] != canonical_roles:
        errors.append(
            f"Target {name} overlay manifest must begin with all six canonical views"
        )
    elif len(overlay_roles) != len(set(overlay_roles)):
        errors.append(f"Target {name} overlay roles must be unique")
    else:
        for record in valid_overlay_records:
            for path_key, digest_key in (
                ("source_render", "source_render_sha256"),
                ("raw_overlay", "raw_overlay_sha256"),
            ):
                artifact = _manifest_run_artifact_path(
                    run_dir,
                    record.get(path_key),
                )
                if artifact is None or not artifact.is_file():
                    errors.append(f"Target {name} overlay has missing {path_key}")
                elif record.get(digest_key) != _sha256_file(artifact):
                    errors.append(
                        f"Target {name} overlay {path_key} digest does not match"
                    )

    if (
        registration.get("schema_version")
        != SEMANTIC_OVERLAY_REGISTRATION_SCHEMA_VERSION
    ):
        errors.append(f"Target {name} registration has an unsupported schema")
    if registration.get("target_semantic_part") != name:
        errors.append(f"Target {name} registration names a different part")
    if registration.get("overlay_manifest_sha256") != _sha256_file(overlay_path):
        errors.append(f"Target {name} registration is not bound to its overlays")
    registration_records = registration.get("records")
    if not isinstance(registration_records, list):
        errors.append(f"Target {name} registration records must be an array")
        valid_registration_records: list[dict[str, object]] = []
    else:
        valid_registration_records = [
            record for record in registration_records if isinstance(record, dict)
        ]
        if len(valid_registration_records) != len(registration_records):
            errors.append(f"Target {name} registration records must all be objects")
    registration_roles = [
        str(record.get("role")) for record in valid_registration_records
    ]
    if registration_roles != overlay_roles:
        errors.append(f"Target {name} registration roles do not match overlays")
        accepted_roles: list[str] = []
    else:
        accepted_roles = [
            str(record["role"])
            for record in valid_registration_records
            if record.get("accepted") is True
        ]
        for record in valid_registration_records:
            if record.get("accepted") is not True:
                continue
            role = str(record.get("role"))
            if (
                not isinstance(record.get("silhouette_iou"), int | float)
                or float(record["silhouette_iou"]) < 0.95
            ):
                errors.append(f"Target {name} accepted view {role} failed IoU")
            if (
                not isinstance(record.get("edge_f_score_2px"), int | float)
                or float(record["edge_f_score_2px"]) < 0.95
            ):
                errors.append(f"Target {name} accepted view {role} failed edge F-score")
            plausibility = record.get("affine_plausibility")
            if (
                not isinstance(plausibility, dict)
                or plausibility.get("accepted") is not True
            ):
                errors.append(
                    f"Target {name} accepted view {role} has implausible affine"
                )
            if (
                not isinstance(record.get("semantic_pixel_count"), int)
                or int(record["semantic_pixel_count"]) <= 0
            ):
                errors.append(f"Target {name} accepted view {role} has an empty mask")
            for path_key, digest_key in (
                ("aligned_overlay", "aligned_overlay_sha256"),
                ("aligned_mask", "aligned_mask_sha256"),
                ("alignment_blend", "alignment_blend_sha256"),
            ):
                artifact = _manifest_run_artifact_path(
                    run_dir,
                    record.get(path_key),
                )
                if artifact is None or not artifact.is_file():
                    errors.append(
                        f"Target {name} accepted view {role} lacks {path_key}"
                    )
                elif record.get(digest_key) != _sha256_file(artifact):
                    errors.append(
                        f"Target {name} accepted view {role} {path_key} "
                        "digest does not match"
                    )
    if registration.get("accepted_view_count") != len(accepted_roles):
        errors.append(f"Target {name} accepted_view_count does not match")
    errors.extend(
        f"Target {name}: {error}"
        for error in _multiview_semantic_role_errors(accepted_roles)
    )

    if projection.get("schema_version") not in {
        MASK_FACE_UNION_SCHEMA_VERSION,
        MASK_FACE_HIERARCHICAL_VOTE_SCHEMA_VERSION,
    }:
        errors.append(f"Target {name} projection has an unsupported schema")
    if projection.get("target_semantic_part") != name:
        errors.append(f"Target {name} projection names a different part")
    if projection.get("registration_manifest_sha256") != _sha256_file(
        registration_path
    ):
        errors.append(f"Target {name} projection is not bound to registration")
    if projection.get("accepted_roles") != accepted_roles:
        errors.append(f"Target {name} projection accepted roles do not match")
    algorithm = projection.get("algorithm")
    if not isinstance(algorithm, dict):
        errors.append(f"Target {name} projection lacks an algorithm contract")
    else:
        union_algorithm = {
            "name": DETERMINISTIC_INITIALIZATION_ALGORITHM,
            "pixel_policy": "every_foreground_mask_pixel",
            "visibility_policy": "closest_ray_intersection_only",
            "multi_view_reduction": "set_union",
            "mesh_adjacency_expansion": "none",
            "mask_morphology": "none",
            "minimum_view_support": 1,
        }
        vote_algorithm = {
            "name": HIERARCHICAL_VOTE_INITIALIZATION_ALGORITHM,
            "ray_pixel_scope": "every_image_pixel",
            "visibility_policy": "closest_ray_intersection_only",
            "positive_pixel_definition": "closest hit at semantic-mask pixel",
            "negative_pixel_definition": "closest hit at non-mask pixel",
            "per_view_positive_rule": ("positive_pixel_votes > negative_pixel_votes"),
            "per_view_negative_rule": ("negative_pixel_votes > positive_pixel_votes"),
            "per_view_tie_rule": "abstain",
            "multi_view_candidate_rule": "union of per-view positive faces",
            "multi_view_accept_rule": "positive_view_votes > negative_view_votes",
            "multi_view_tie_rule": "reject",
            "view_weighting": "one equal vote per visible view",
            "mesh_adjacency_expansion": "none",
            "mask_morphology": "none",
        }
        expected = (
            vote_algorithm
            if projection.get("schema_version")
            == MASK_FACE_HIERARCHICAL_VOTE_SCHEMA_VERSION
            else union_algorithm
        )
        if algorithm != expected:
            errors.append(f"Target {name} projection algorithm does not match")
    return errors


def _validate_live_lock_candidate(
    *,
    run_dir: Path,
    parts: list[dict[str, object]],
    locks: list[dict[str, object]],
    index: int,
    other_segment_id: int,
) -> list[str]:
    part = parts[index]
    lock = locks[index]
    name = str(part.get("name", part.get("part_name")))
    segment_id = int(part["segment_id"])
    errors: list[str] = []
    if lock.get("part_name") != name:
        errors.append(f"lock part_name must be {name!r}")
    if lock.get("segment_id") != segment_id:
        errors.append(f"lock segment_id must be {segment_id}")
    if lock.get("order") != index:
        errors.append(f"lock order must be {index}")
    if lock.get("unresolved_issues") != []:
        errors.append("unresolved_issues must be an empty list")

    validation_artifacts = lock.get("validation_artifacts")
    if not isinstance(validation_artifacts, list) or not validation_artifacts:
        errors.append("validation_artifacts must cite at least one artifact")
    else:
        for relative in validation_artifacts:
            artifact_path = _safe_run_artifact_path(run_dir, relative)
            if artifact_path is None or not artifact_path.is_file():
                errors.append(f"validation artifact is missing or unsafe: {relative!r}")

    errors.extend(
        validate_part_lock_selection_evidence(
            run_dir,
            part=part,
            lock=lock,
        )
    )
    errors.extend(
        _validate_part_semantic_overlay_evidence(
            run_dir,
            part=part,
            lock=lock,
        )
    )
    for key, label in (
        ("focused_render_dir", "focused render directory"),
        ("canonical_render_dir", "canonical render directory"),
    ):
        raw_path = lock.get(key)
        candidate = (
            _safe_run_artifact_path(run_dir, raw_path)
            if isinstance(raw_path, str)
            else None
        )
        if candidate is None or not candidate.is_dir():
            errors.append(f"{label} is missing or unsafe: {raw_path!r}")
    completion_path = _safe_run_artifact_path(
        run_dir,
        lock.get("semantic_completion"),
    )
    if completion_path is None or not completion_path.is_file():
        errors.append("semantic_completion is missing or unsafe")
    else:
        completion, completion_error = _load_json_object(
            completion_path,
            label=f"{name} semantic completion",
        )
        if completion_error:
            errors.append(completion_error)
        elif completion is not None and completion.get("status") != "complete":
            errors.append("semantic_completion status must be 'complete'")

    labels_path = _safe_run_artifact_path(run_dir, lock.get("face_labels"))
    labels: array[int] | None = None
    if labels_path is None:
        errors.append("face_labels must be a safe run-relative path")
    else:
        labels, labels_error = _load_u32_labels(labels_path)
        if labels_error:
            errors.append(labels_error)
        else:
            assert labels is not None
            digest = _sha256_file(labels_path)
            if lock.get("face_labels_sha256") != digest:
                errors.append("face_labels_sha256 does not match the label file")
            accepted_face_count = sum(value == segment_id for value in labels)
            if accepted_face_count <= 0:
                errors.append("accepted face set is empty")
            if lock.get("accepted_face_count") != accepted_face_count:
                errors.append("accepted_face_count does not match the label file")

    if labels is not None:
        allowed_ids = {
            other_segment_id,
            *(int(item["segment_id"]) for item in parts[: index + 1]),
        }
        unexpected_ids = set(labels) - allowed_ids
        if unexpected_ids:
            errors.append(
                f"label file contains unstarted segment IDs {sorted(unexpected_ids)}"
            )
        for prior_index, prior_lock in enumerate(locks[:index]):
            prior_path = _safe_run_artifact_path(
                run_dir,
                prior_lock.get("face_labels"),
            )
            if prior_path is None:
                errors.append(f"prior lock {prior_index} has an unsafe label path")
                continue
            prior_labels, prior_error = _load_u32_labels(prior_path)
            if prior_error:
                errors.append(prior_error)
                continue
            assert prior_labels is not None
            if len(prior_labels) != len(labels):
                errors.append("source face count changed after a prior lock")
                continue
            prior_id = int(parts[prior_index]["segment_id"])
            if any(
                prior == prior_id and labels[face_index] != prior_id
                for face_index, prior in enumerate(prior_labels)
            ):
                prior_name = parts[prior_index].get(
                    "name",
                    parts[prior_index].get("part_name"),
                )
                errors.append(f"previously locked part {prior_name!r} changed")
        if index == len(parts) - 1 and other_segment_id in labels:
            errors.append("body fallback has not consumed every residual other face")
    return errors


def _record_structural_approval(
    run_dir: Path,
    *,
    records: list[dict[str, object]],
    lock_obj: threading.Lock,
    part: dict[str, object],
    lock: dict[str, object],
    fingerprint: str,
) -> None:
    """Record that the launcher observed one lock live, without a vision claim.

    Terminal validation binds each lock to a passing review by
    (part_name, lock_fingerprint, face_labels_sha256). That binding is what
    stops a corpus assembled at the end from passing; it does not depend on the
    verdict having come from a vision model. `review_kind` keeps the record
    honest about which check actually ran.
    """

    record = {
        "part_name": part.get("name", part.get("part_name")),
        "lock_fingerprint": fingerprint,
        "face_labels_sha256": lock.get("face_labels_sha256"),
        "status": "passed",
        "review_kind": "structural_sequential_gate",
        "errors": [],
    }
    with lock_obj:
        records.append(record)
        _write_json(
            run_dir / "part_review_manifest.json",
            {
                "schema_version": "content-agents.mesh-segmentation-part-reviews.v1",
                "reviews": list(records),
            },
        )


def _make_recognition_sequential_watchdog(
    run_dir: Path,
    *,
    review_candidate: (
        Callable[
            [dict[str, object], dict[str, object], str],
            tuple[bool, list[str], dict[str, object]],
        ]
        | None
    ) = None,
) -> Callable[[], WatchdogFailure | None]:
    """Create a trusted evidence and vision gate for sequential recognition."""

    run_dir.mkdir(parents=True, exist_ok=True)
    gate_path = run_dir / "live_sequential_gate.json"
    approved_fingerprints: list[str] = []
    approved_parts: list[str] = []
    queue_first_seen_mtime: int | None = None
    review_lock = threading.Lock()
    review_state: dict[str, object] = {
        "fingerprint": None,
        "status": None,
        "errors": [],
        "record": None,
    }
    review_records: list[dict[str, object]] = []

    def start_review(
        *,
        part: dict[str, object],
        lock: dict[str, object],
        fingerprint: str,
    ) -> None:
        with review_lock:
            review_state.update(
                {
                    "fingerprint": fingerprint,
                    "status": "running",
                    "errors": [],
                    "record": None,
                }
            )

        def worker() -> None:
            try:
                if review_candidate is None:
                    raise RuntimeError("Independent per-part reviewer is unavailable")
                passed, errors, record = review_candidate(part, lock, fingerprint)
            except Exception as exc:  # noqa: BLE001 - surface review failure to child
                passed = False
                errors = [
                    "Independent per-part VQA could not complete: "
                    f"{type(exc).__name__}: {exc}"
                ]
                record = {
                    "part_name": part.get("name", part.get("part_name")),
                    "lock_fingerprint": fingerprint,
                    "status": "review_error",
                    "errors": errors,
                }
            with review_lock:
                review_records.append(record)
                _write_json(
                    run_dir / "part_review_manifest.json",
                    {
                        "schema_version": (
                            "content-agents.mesh-segmentation-part-reviews.v1"
                        ),
                        "reviews": list(review_records),
                    },
                )
                if review_state.get("fingerprint") == fingerprint:
                    review_state.update(
                        {
                            "status": "passed" if passed else "failed",
                            "errors": list(errors),
                            "record": record,
                        }
                    )

        threading.Thread(
            target=worker,
            name=f"mesh-segmentation-review-{fingerprint[:8]}",
            daemon=True,
        ).start()

    _write_live_sequential_gate(
        gate_path,
        status="waiting_for_queue",
        approved_parts=[],
        authorized_segment_ids=[0],
        next_part=None,
    )

    def reject(reason: str) -> WatchdogFailure:
        _write_live_sequential_gate(
            gate_path,
            status="rejected",
            approved_parts=list(approved_parts),
            authorized_segment_ids=[0],
            next_part=None,
            problems=[reason],
        )
        return WatchdogFailure(
            f"Sequential mesh-segmentation gate rejected the run: {reason}",
            fatal=True,
        )

    def watchdog() -> WatchdogFailure | None:
        nonlocal queue_first_seen_mtime
        queue_path = run_dir / "part_work_queue.json"
        hypothesis_candidates = [
            *(run_dir / "hypotheses").glob("**/face_labels.u32le"),
            *(run_dir / "edits").glob("**/face_labels.u32le"),
            *(run_dir / "final").glob("face_labels.u32le"),
        ]
        hypothesis_snapshots: list[tuple[int, str, Path]] = []
        for path in hypothesis_candidates:
            try:
                mtime_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                # The child may replace a revision atomically while the
                # watchdog snapshots it. A vanished candidate is observed on
                # the next pass instead of crashing this pass.
                continue
            hypothesis_snapshots.append((mtime_ns, str(path), path))
        hypothesis_snapshots.sort()
        observed_ids: set[int] = set()
        earliest_semantic_mtime: int | None = None
        for labels_mtime, _labels_name, labels_path in hypothesis_snapshots:
            labels, labels_error = _load_u32_labels(labels_path)
            if labels_error:
                return WatchdogFailure(labels_error, fatal=False)
            assert labels is not None
            nonzero_ids = {value for value in labels if value != 0}
            if nonzero_ids:
                observed_ids.update(nonzero_ids)
                earliest_semantic_mtime = (
                    labels_mtime
                    if earliest_semantic_mtime is None
                    else min(earliest_semantic_mtime, labels_mtime)
                )

        if not queue_path.is_file():
            if observed_ids:
                return reject(
                    "semantic face labels appeared before part_work_queue.json"
                )
            _write_live_sequential_gate(
                gate_path,
                status="waiting_for_queue",
                approved_parts=list(approved_parts),
                authorized_segment_ids=[0],
                next_part=None,
            )
            return None

        def queue_problem(reason: str) -> WatchdogFailure | None:
            if observed_ids:
                return reject(reason)
            _write_live_sequential_gate(
                gate_path,
                status="waiting_for_valid_queue",
                approved_parts=list(approved_parts),
                authorized_segment_ids=[0],
                next_part=None,
                problems=[reason],
            )
            return None

        queue, queue_error = _load_json_object(
            queue_path,
            label="part_work_queue.json",
        )
        if queue_error:
            return WatchdogFailure(queue_error, fatal=False)
        assert queue is not None
        if queue.get("other_segment_id") != 0:
            return queue_problem("part_work_queue.json must reserve other_segment_id 0")
        raw_parts = queue.get("parts")
        if not isinstance(raw_parts, list) or not raw_parts:
            problem = (
                "part_work_queue.json must contain the ordered entries in a "
                "non-empty `parts` list"
            )
            if observed_ids:
                return reject(problem)
            return queue_problem(problem)
        if not all(isinstance(part, dict) for part in raw_parts):
            return queue_problem("every recognition queue part must be an object")
        parts: list[dict[str, object]] = list(raw_parts)

        names: list[str] = []
        segment_ids: list[int] = []
        scope_ranks: list[int] = []
        body_indices: list[int] = []
        for index, part in enumerate(parts):
            name = part.get("name", part.get("part_name"))
            segment_id = part.get("segment_id")
            role = part.get("role")
            scope = part.get("estimated_scope")
            name_error = _semantic_part_name_error(name)
            if name_error:
                return queue_problem(f"queue part {index} name {name!r} {name_error}")
            assert isinstance(name, str)
            if not isinstance(segment_id, int) or segment_id <= 0:
                return queue_problem(f"queue part {name!r} has an invalid segment_id")
            if role not in {"distinctive", "body_fallback"}:
                return queue_problem(f"queue part {name!r} has invalid role {role!r}")
            if scope not in RECOGNITION_SCOPE_ORDER:
                return queue_problem(
                    f"queue part {name!r} has invalid estimated_scope {scope!r}"
                )
            if role == "body_fallback":
                body_indices.append(index)
                if scope != "body":
                    return queue_problem(
                        f"body fallback {name!r} must have estimated_scope 'body'"
                    )
            names.append(name)
            segment_ids.append(segment_id)
            scope_ranks.append(RECOGNITION_SCOPE_ORDER[str(scope)])
        if len(set(names)) != len(names) or len(set(segment_ids)) != len(segment_ids):
            return queue_problem("queue part names and segment IDs must be unique")
        if body_indices != [len(parts) - 1]:
            return queue_problem("the single body_fallback must be the last queue part")
        distinctive_scope_ranks = scope_ranks[:-1]
        if any(
            left < right
            for left, right in zip(
                distinctive_scope_ranks,
                distinctive_scope_ranks[1:],
                strict=False,
            )
        ):
            return queue_problem(
                "queue distinctive estimated_scope order must be large-to-small"
            )
        if queue_first_seen_mtime is None:
            queue_first_seen_mtime = queue_path.stat().st_mtime_ns
            if (
                earliest_semantic_mtime is not None
                and queue_first_seen_mtime > earliest_semantic_mtime
            ):
                return reject("part_work_queue.json was written after semantic labels")

        approved_count = len(approved_fingerprints)
        if approved_count > len(parts):
            return reject("trusted gate state exceeds the current queue length")
        approved_ids = set(segment_ids[:approved_count])
        active_ids = (
            {segment_ids[approved_count]} if approved_count < len(parts) else set()
        )
        unexpected_observed_ids = observed_ids - {0, *approved_ids, *active_ids}
        if unexpected_observed_ids:
            return reject(
                "a later part was labeled before the current part was approved: "
                f"{sorted(unexpected_observed_ids)}"
            )
        unknown_ids = observed_ids - set(segment_ids)
        if unknown_ids:
            return reject(
                f"hypotheses contain unknown segment IDs {sorted(unknown_ids)}"
            )

        lock_path = run_dir / "part_lock_manifest.json"
        locks: list[dict[str, object]] = []
        if lock_path.is_file():
            manifest, manifest_error = _load_json_object(
                lock_path,
                label="part_lock_manifest.json",
            )
            if manifest_error:
                return WatchdogFailure(manifest_error, fatal=False)
            assert manifest is not None
            raw_locks = manifest.get("locks")
            if not isinstance(raw_locks, list) or not all(
                isinstance(lock, dict) for lock in raw_locks
            ):
                return reject("part_lock_manifest.json must contain an object array")
            locks = list(raw_locks)
        if len(locks) > approved_count + 1:
            return reject("multiple unapproved part locks were written at once")

        for index, fingerprint in enumerate(approved_fingerprints):
            if index >= len(locks):
                return reject("an approved part lock was removed")
            current_fingerprint = hashlib.sha256(
                json.dumps(locks[index], sort_keys=True).encode("utf-8")
            ).hexdigest()
            if current_fingerprint != fingerprint:
                return reject(f"approved lock {index} was mutated")
            approved_labels_path = _safe_run_artifact_path(
                run_dir,
                locks[index].get("face_labels"),
            )
            if (
                approved_labels_path is None
                or not approved_labels_path.is_file()
                or _sha256_file(approved_labels_path)
                != locks[index].get("face_labels_sha256")
            ):
                return reject(f"approved lock {index} label snapshot was mutated")

        for index, part in enumerate(parts):
            status = part.get("status")
            if index < approved_count and status != "locked":
                return reject(f"approved queue part {names[index]!r} is not locked")
            if index > approved_count and status == "locked":
                return reject(f"future queue part {names[index]!r} is pre-locked")
        active_part = queue.get("active_part")
        if approved_count < len(parts):
            current_name = names[approved_count]
            previous_name = names[approved_count - 1] if approved_count else None
            candidate_present = len(locks) == approved_count + 1
            current_status = parts[approved_count].get("status")
            if current_status == "locked" and not candidate_present:
                return reject(f"queue part {current_name!r} locked without evidence")
            if active_part not in {None, current_name, previous_name}:
                return reject(
                    f"active_part must be the current queue part {current_name!r}"
                )
            if (
                previous_name is not None
                and active_part == previous_name
                and segment_ids[approved_count] in observed_ids
            ):
                return reject(
                    f"segment {segment_ids[approved_count]} appeared before "
                    f"active_part advanced to {current_name!r}"
                )
            if candidate_present and current_status == "locked":
                candidate_errors = _validate_live_lock_candidate(
                    run_dir=run_dir,
                    parts=parts,
                    locks=locks,
                    index=approved_count,
                    other_segment_id=0,
                )
                if not candidate_errors:
                    fingerprint = hashlib.sha256(
                        json.dumps(locks[approved_count], sort_keys=True).encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    if review_candidate is None:
                        # The independent visual reviewer is retired. This gate
                        # still proves the launcher observed this exact lock
                        # live, which is the anti-backfill property terminal
                        # validation binds to; it simply no longer carries a
                        # vision verdict. Record that honestly rather than
                        # failing every recognition run for a review nothing
                        # can produce.
                        _record_structural_approval(
                            run_dir,
                            records=review_records,
                            lock_obj=review_lock,
                            part=parts[approved_count],
                            lock=locks[approved_count],
                            fingerprint=fingerprint,
                        )
                        reviewed_fingerprint = fingerprint
                        review_status = "passed"
                        review_errors = []
                    else:
                        with review_lock:
                            reviewed_fingerprint = review_state.get("fingerprint")
                            review_status = review_state.get("status")
                            review_errors = list(review_state.get("errors", []))
                    if reviewed_fingerprint != fingerprint:
                        start_review(
                            part=parts[approved_count],
                            lock=locks[approved_count],
                            fingerprint=fingerprint,
                        )
                        _write_live_sequential_gate(
                            gate_path,
                            status="independent_review_in_progress",
                            approved_parts=list(approved_parts),
                            authorized_segment_ids=[
                                0,
                                *segment_ids[: approved_count + 1],
                            ],
                            next_part=current_name,
                            problems=[
                                "The launcher is independently reviewing this "
                                "part's focused and canonical pixels."
                            ],
                        )
                        return None
                    if review_status == "running":
                        _write_live_sequential_gate(
                            gate_path,
                            status="independent_review_in_progress",
                            approved_parts=list(approved_parts),
                            authorized_segment_ids=[
                                0,
                                *segment_ids[: approved_count + 1],
                            ],
                            next_part=current_name,
                        )
                        return None
                    if review_status != "passed":
                        _write_live_sequential_gate(
                            gate_path,
                            status="independent_review_failed",
                            approved_parts=list(approved_parts),
                            authorized_segment_ids=[
                                0,
                                *segment_ids[: approved_count + 1],
                            ],
                            next_part=current_name,
                            problems=review_errors
                            or ["Independent visual review rejected the part lock"],
                        )
                        return None
                    approved_fingerprints.append(fingerprint)
                    approved_parts.append(current_name)
                    approved_count += 1
                else:
                    _write_live_sequential_gate(
                        gate_path,
                        status="waiting_for_valid_lock",
                        approved_parts=list(approved_parts),
                        authorized_segment_ids=[
                            0,
                            *segment_ids[: min(approved_count + 1, len(parts))],
                        ],
                        next_part=current_name,
                        problems=candidate_errors,
                    )
                    return None

        next_part = names[approved_count] if approved_count < len(parts) else None
        status = "complete" if next_part is None else "approved_for_next_part"
        _write_live_sequential_gate(
            gate_path,
            status=status,
            approved_parts=list(approved_parts),
            authorized_segment_ids=[
                0,
                *segment_ids[: min(approved_count + 1, len(parts))],
            ],
            next_part=next_part,
        )
        return None

    return watchdog


@dataclass(frozen=True)
class OutstandingWork:
    """What a turn left unfinished, and how much of it."""

    description: str
    unlocked_parts: int
    missing_artifacts: int
    progress_units: int = 0

    @property
    def counts(self) -> tuple[int, int, int]:
        """The values the loop compares to detect a turn that changed nothing."""
        return (self.unlocked_parts, self.missing_artifacts, self.progress_units)


def _outstanding_made_durable_progress(
    before: OutstandingWork | None,
    after: OutstandingWork | None,
) -> bool:
    """Whether a timed-out turn durably reduced or completed outstanding work."""

    if before is None:
        return False
    if after is None:
        return True
    return (
        after.unlocked_parts < before.unlocked_parts
        or after.progress_units > before.progress_units
    )


def _outstanding_progress_units(
    outstanding: OutstandingWork | None, *, target_count: int
) -> int:
    """Report targeted locks or recognition locks without conflating the modes."""

    if outstanding is None:
        return target_count
    return max(target_count - outstanding.unlocked_parts, outstanding.progress_units)


def _recognition_lock_progress(run_dir: Path) -> int:
    """Count durable recognition locks without replacing terminal validation."""

    lock_counts: list[int] = []
    for relative, collection, name_key in (
        ("part_work_queue.json", "parts", "name"),
        ("part_lock_manifest.json", "locks", "part_name"),
    ):
        try:
            payload = json.loads((run_dir / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if relative == "part_work_queue.json" and payload.get("mode") != "recognition":
            continue
        records = payload.get(collection)
        if not isinstance(records, list):
            continue
        locked_parts: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            if relative == "part_work_queue.json" and record.get("status") != "locked":
                continue
            part_name = record.get(name_key)
            if isinstance(part_name, str) and part_name.strip():
                locked_parts.add(part_name.strip())
        lock_counts.append(len(locked_parts))
    return max(lock_counts, default=0)


def _outstanding_work(
    run_dir: Path, *, config: MeshSegmentationConfig
) -> OutstandingWork | None:
    """Describe work the child left unfinished, or None when the run is done.

    Used only to decide whether another turn is worth starting. It is
    deliberately cheap and permissive: terminal validation remains the
    authority on whether a finished run is actually valid.
    """

    required = list(REQUIRED_FINAL_ARTIFACTS)
    if config.target_semantic_parts:
        required += list(REQUIRED_TARGETED_FINAL_ARTIFACTS)
    else:
        # Terminal validation demands these of every recognition run. Omitting
        # them here declared such a run finished as soon as the shared base
        # artifacts existed, so the loop stopped granting turns and the run
        # then failed validation for artifacts it was never given a chance to
        # write.
        required += list(REQUIRED_RECOGNITION_FINAL_ARTIFACTS)
    missing = [name for name in required if not (run_dir / name).is_file()]

    unlocked: list[str] = []
    recognition_locks = 0
    if config.target_semantic_parts:
        directories = _part_directories(run_dir)
        for target in config.target_semantic_parts:
            part_dir = directories.get(_normalize_part_key(target))
            completion: object = None
            if part_dir is not None:
                try:
                    completion = json.loads(
                        (part_dir / "part_completion.json").read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    pass
            # A completion artifact is also how an agent honestly records a
            # provisional or deferred result.  Presence alone therefore says
            # nothing about completion: only the terminal state can stop the
            # continuation loop.  Terminal validation remains responsible for
            # checking the lock's evidence and digests in full.
            if not isinstance(completion, dict) or completion.get("status") != "locked":
                unlocked.append(target)
    else:
        recognition_locks = _recognition_lock_progress(run_dir)

    if not missing and not unlocked:
        return None
    details: list[str] = []
    if unlocked:
        details.append(
            f"{len(unlocked)} of {len(config.target_semantic_parts)} target parts "
            f"unlocked ({', '.join(unlocked[:4])}"
            + ("..." if len(unlocked) > 4 else "")
            + ")"
        )
    if missing:
        details.append(f"{len(missing)} required artifacts missing")
    if recognition_locks:
        details.append(f"{recognition_locks} recognition parts durably locked")
    return OutstandingWork(
        description="; ".join(details),
        unlocked_parts=len(unlocked),
        missing_artifacts=len(missing),
        progress_units=recognition_locks,
    )


def _continuation_prompt(
    prompt: str,
    run_dir: Path,
    *,
    attempt: int,
    config: MeshSegmentationConfig,
) -> str:
    """Prefix the frozen prompt with the state a resuming turn must respect."""

    outstanding = _outstanding_work(run_dir, config=config)
    summary = outstanding.description if outstanding else "final artifacts"
    locked_revision_guidance = (
        "Accepted locks are face-immutable in every run mode: never extend, "
        "reassign, or otherwise change a face that belongs to an accepted lock. "
        "Continue only from unlocked, provisional, or residual faces. "
    )
    return (
        f"Continuation turn {attempt}. A previous turn on this same run "
        f"directory ended before the work was complete: {summary}.\n\n"
        "Read the run directory first and resume from what is already there. "
        "Every accepted lock is stable by default, and its revision and evidence "
        "history are immutable: never rewrite or delete them. "
        f"{locked_revision_guidance}"
        "Continue with "
        "the parts that are not yet locked, including any part whose "
        "part_completion.json says provisional or deferred, then produce the "
        "remaining final artifacts. An existing part_deferral.json does not "
        "finish a requested target: reopen that target. Treat each requested "
        "target as a positive "
        "assertion and, before deferring again, audit every still-unlocked "
        "topology component using the exhaustive isolated-geometry rule in the "
        "skill. Record every component disposition. If a part still cannot be "
        "locked after that genuine effort, say so explicitly in the final "
        "report rather than silently stopping.\n\n"
        f"{prompt}"
    )


_TURN_USAGE_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
)

CUMULATIVE_USAGE_SCHEMA_VERSION = "content-agents.mesh-segmentation-run-usage.v1"
CHILD_TURN_TIMEOUT_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-child-turn-timeout.v1"
)


def _record_turn_usage(
    run_dir: Path,
    *,
    bridge_artifact_prefix: str,
    attempt: int,
    totals: dict[str, int],
) -> None:
    """Fold one turn's usage into the run total and preserve its artifacts.

    The bridge unlinks and recreates its result artifact on every turn, so
    after an N-turn run the file describes only turn N. Reading it directly
    understates a multi-turn run's cost by up to the iteration budget. Fold
    each turn's usage into a cumulative artifact as the turn completes, and
    keep the per-turn result and child log so a run stays diagnosable.
    """

    raw_dir = run_dir / "raw"
    result_path = raw_dir / f"{bridge_artifact_prefix}_result.json"
    if not result_path.is_file():
        return
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(result, dict):
        return

    usage = result.get("usage")
    if isinstance(usage, dict):
        for key in _TURN_USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            totals[key] = totals.get(key, 0) + int(value)
    items = result.get("items")
    if isinstance(items, list):
        commands = [
            item
            for item in items
            if isinstance(item, dict) and item.get("type") == "command_execution"
        ]
        totals["command_calls"] = totals.get("command_calls", 0) + len(commands)
        totals["failed_command_calls"] = totals.get("failed_command_calls", 0) + sum(
            item.get("status") == "failed"
            or (item.get("exit_code") is not None and str(item.get("exit_code")) != "0")
            for item in commands
        )
        totals["model_turn_count"] = totals.get("model_turn_count", 0) + sum(
            isinstance(item, dict) and item.get("type") == "agent_message"
            for item in items
        )
    totals["child_turn_count"] = attempt

    # Keep this turn's evidence before the next turn's bridge run replaces it.
    for source in (result_path, raw_dir / f"{bridge_artifact_prefix}_items.json"):
        if source.is_file():
            shutil.copy2(source, source.with_name(f"{source.stem}.turn-{attempt}.json"))
    child_log = run_dir / "child-output.log"
    if child_log.is_file():
        shutil.copy2(child_log, run_dir / f"child-output.turn-{attempt}.log")

    payload: dict[str, object] = {
        "schema_version": CUMULATIVE_USAGE_SCHEMA_VERSION,
        **{key: totals[key] for key in sorted(totals)},
    }
    payload["total_tokens"] = totals.get("input_tokens", 0) + totals.get(
        "output_tokens", 0
    )
    _write_json(raw_dir / f"{bridge_artifact_prefix}_usage_total.json", payload)


def _record_child_turn_timeout(
    run_dir: Path,
    *,
    attempt: int,
    timeout_seconds: float,
    locked_before: int,
    locked_after: int,
    semantic_progress_detected: bool,
    progress_stage: str | None = None,
    progress_summary: str | None = None,
) -> None:
    """Preserve the timed-out turn before a fresh bridge run replaces its log."""

    child_log = run_dir / "child-output.log"
    if child_log.is_file():
        shutil.copy2(child_log, run_dir / f"child-output.turn-{attempt}.log")
    _write_json(
        run_dir / "raw" / f"child_turn_timeout.turn-{attempt}.json",
        {
            "schema_version": CHILD_TURN_TIMEOUT_SCHEMA_VERSION,
            "observed_at": utc_now(),
            "turn": attempt,
            "timeout_seconds": timeout_seconds,
            "locked_target_count_before": locked_before,
            "locked_target_count_after": locked_after,
            "semantic_progress_detected": semantic_progress_detected,
            "last_observed_stage": progress_stage,
            "last_observed_progress": progress_summary,
        },
    )


_MESH_PROGRESS_STAGES = (
    "child_running",
    "planning",
    "preparing",
    "segmenting",
    "validating",
    "finalizing",
)


def _regular_progress_artifact(run_dir: Path, relative: str) -> Path | None:
    path = run_dir / relative
    try:
        resolved_root = run_dir.resolve(strict=True)
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    expected = resolved_root / relative
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or resolved != expected
    ):
        return None
    return path


def _observe_mesh_segmentation_progress(run_dir: Path) -> tuple[str, list[Path]]:
    """Infer only coarse, evidence-backed stages from durable run artifacts."""

    stage = "child_running"
    artifacts: list[Path] = []
    try:
        run_root = run_dir.resolve(strict=True)
    except OSError:
        return stage, artifacts

    def observe(candidate_stage: str, *relative_paths: str) -> None:
        nonlocal artifacts, stage
        present = [
            path
            for relative in relative_paths
            if (path := _regular_progress_artifact(run_dir, relative)) is not None
        ]
        if present and _MESH_PROGRESS_STAGES.index(candidate_stage) > (
            _MESH_PROGRESS_STAGES.index(stage)
        ):
            stage = candidate_stage
            artifacts = present

    observe("planning", "part_plan.json", "part_work_queue.json")
    observe(
        "preparing",
        "prepare/topology.json",
        "fragments/fragment_manifest.json",
    )
    try:
        part_directories = tuple(_part_directories(run_root).values())
    except OSError:
        part_directories = ()
    revision_artifacts = [
        path
        for part_dir in part_directories
        for relative in ("initializer_decision.json", "rev-000/face_labels.u32le")
        if (
            path := _regular_progress_artifact(
                run_dir, str((part_dir / relative).relative_to(run_root))
            )
        )
        is not None
    ]
    if revision_artifacts:
        stage = "segmenting"
        artifacts = revision_artifacts
    validation_artifacts = [
        path
        for part_dir in part_directories
        for relative in ("part_completion.json", "falsification_validation.json")
        if (
            path := _regular_progress_artifact(
                run_dir, str((part_dir / relative).relative_to(run_root))
            )
        )
        is not None
    ]
    validation_artifacts.extend(
        path
        for relative in ("part_lock_manifest.json", "part_review_manifest.json")
        if (path := _regular_progress_artifact(run_dir, relative)) is not None
    )
    if validation_artifacts:
        stage = "validating"
        artifacts = validation_artifacts
    observe(
        "finalizing",
        "state/final_labels.u32le",
        "final/segmented.usdc",
        "final/export_manifest.json",
        "final/report.md",
    )
    return stage, artifacts


@dataclass
class MeshSegmentationProgressTracker:
    """Publish bounded parent-owned transitions without fabricated percentages."""

    run_dir: Path
    trace_writer: TraceWriter
    last_stage: str = "child_running"
    last_summary: str = "Mesh-segmentation child agent is running."
    _started: bool = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.trace_writer.write(
            "mesh_segmentation_progress",
            phase="runner",
            summary=self.last_summary,
            data={"stage": self.last_stage},
        )

    def poll(self) -> WatchdogFailure | None:
        self.start()
        try:
            stage, artifacts = _observe_mesh_segmentation_progress(self.run_dir)
        except Exception as exc:  # noqa: BLE001 - telemetry must not stop the child
            LOGGER.debug("Unable to observe mesh-segmentation progress: %s", exc)
            return None
        if _MESH_PROGRESS_STAGES.index(stage) <= _MESH_PROGRESS_STAGES.index(
            self.last_stage
        ):
            return None
        self.last_stage = stage
        self.last_summary = f"Mesh-segmentation advanced to {stage}."
        self.trace_writer.write(
            "mesh_segmentation_progress",
            phase="runner",
            summary=self.last_summary,
            artifacts=[str(path) for path in artifacts],
            data={"stage": stage},
        )
        return None


def _make_agent_memory_checkpoint_watchdog(
    run_dir: Path,
    *,
    target_parts: tuple[str, ...],
    memory: AgentMemory | None,
    remember: Callable[[RememberRequest], object] | None = None,
    target_prim_path: str | None = None,
) -> Callable[[], WatchdogFailure | None] | None:
    """Checkpoint exact revision evidence before a later correction can replace it."""

    if memory is None:
        return None
    if remember is None:
        raise ValueError(
            "Agent-memory checkpointing requires the launcher-owned broker"
        )
    record_observation = remember
    captured: set[tuple[str, str, str]] = set()
    reported_failures: set[tuple[str, str, str]] = set()

    def regular_file_metadata(path: Path) -> os.stat_result | None:
        try:
            metadata = path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return None
        return metadata

    def discover_part_names() -> tuple[str, ...]:
        names = list(target_parts)
        for artifact_name in ("part_work_queue.json", "part_plan.json"):
            payload, error = _load_json_object(
                run_dir / artifact_name,
                label=artifact_name,
            )
            if error or payload is None:
                continue
            raw_parts = payload.get("parts")
            if not isinstance(raw_parts, list):
                continue
            for part in raw_parts:
                if not isinstance(part, dict):
                    continue
                name = part.get("name", part.get("part_name"))
                if _semantic_part_name_error(name) is None:
                    assert isinstance(name, str)
                    names.append(name)
        return tuple(dict.fromkeys(names))

    def checkpoint_revision(
        part_name: str,
        part_dir: Path,
        revision_dir: Path,
    ) -> None:
        labels_path = revision_dir / "face_labels.u32le"
        # Revisions are assembled through several atomic file writes. The
        # directory can become visible before the candidate labels do.
        labels_metadata = regular_file_metadata(labels_path)
        if labels_metadata is None:
            return
        candidate_time_ns = labels_metadata.st_mtime_ns
        labels_digest = _sha256_file(labels_path)
        identity = (part_name, revision_dir.name, labels_digest)
        if identity in captured:
            return

        def bound_frontier(path: Path) -> bool:
            if regular_file_metadata(path) is None:
                return False
            payload, error = _load_json_object(path, label="frontier audit")
            if error or payload is None:
                return False
            raw_labels = payload.get("face_labels")
            if not isinstance(raw_labels, str):
                return False
            declared_labels = Path(raw_labels)
            if not declared_labels.is_absolute():
                declared_labels = path.parent / declared_labels
            return (
                Path(os.path.abspath(declared_labels))
                == Path(os.path.abspath(labels_path))
                and payload.get("face_labels_sha256") == labels_digest
            )

        def bound_render_images(render_dir: Path) -> list[Path] | None:
            manifest_path = render_dir / "render_manifest.json"
            if regular_file_metadata(manifest_path) is None:
                return None
            payload, error = _load_json_object(
                manifest_path,
                label="selected-only render manifest",
            )
            if error or payload is None:
                return None
            raw_scene = payload.get("scene")
            if not isinstance(raw_scene, str):
                return None
            scene_path = Path(raw_scene)
            if not scene_path.is_absolute():
                scene_path = manifest_path.parent / scene_path
            scene_path = Path(os.path.abspath(scene_path))
            revision_root = Path(os.path.abspath(revision_dir))
            try:
                scene_path.relative_to(revision_root)
            except ValueError:
                return None
            if regular_file_metadata(scene_path) is None:
                return None
            try:
                scene_digest = _sha256_file(scene_path)
            except OSError:
                return None
            if payload.get("scene_sha256") != scene_digest:
                return None
            raw_renders = payload.get("renders")
            if not isinstance(raw_renders, list):
                return None
            images: list[Path] = []
            for record in raw_renders:
                if not isinstance(record, dict):
                    continue
                raw_image = record.get("image")
                if not isinstance(raw_image, str):
                    continue
                image_path = Path(raw_image)
                if not image_path.is_absolute():
                    image_path = manifest_path.parent / image_path
                image_path = Path(os.path.abspath(image_path))
                try:
                    image_path.relative_to(Path(os.path.abspath(render_dir)))
                    if regular_file_metadata(image_path) is None:
                        continue
                    image_digest = _sha256_file(image_path)
                except (OSError, ValueError):
                    continue
                if record.get("image_sha256") == image_digest:
                    images.append(image_path)
            return sorted(set(images))

        selected_render_candidates = (
            revision_dir / "selected-only" / "renders",
            part_dir / "review" / "selected-only" / f"renders-{revision_dir.name}",
            part_dir / "review" / "selected-only" / "renders",
            part_dir / "selected-only" / "renders",
        )
        frontier_candidates = {
            revision_dir / "frontier-audit" / "frontier_audit.json",
            revision_dir / "frontier" / "frontier_audit.json",
            part_dir / "review" / "frontier_audit.json",
            *part_dir.glob("review/frontier*/frontier_audit.json"),
            *part_dir.glob("frontier*/frontier_audit.json"),
        }
        selected_render_dir: Path | None = None
        selected_images: list[Path] = []
        for path in selected_render_candidates:
            manifest_path = path / "render_manifest.json"
            manifest_metadata = regular_file_metadata(manifest_path)
            if manifest_metadata is None:
                continue
            is_fresh = manifest_metadata.st_mtime_ns > candidate_time_ns
            bound_images = bound_render_images(path) if is_fresh else None
            if bound_images is not None:
                selected_render_dir = path
                selected_images = bound_images
                break
        render_manifest_path = (
            selected_render_dir / "render_manifest.json"
            if selected_render_dir is not None
            else None
        )
        fresh_frontiers: list[tuple[int, Path]] = []
        for path in frontier_candidates:
            frontier_metadata = regular_file_metadata(path)
            if frontier_metadata is None:
                continue
            modified_at_ns = frontier_metadata.st_mtime_ns
            if modified_at_ns > candidate_time_ns and bound_frontier(path):
                fresh_frontiers.append((modified_at_ns, path))
        frontier_path = (
            max(fresh_frontiers, key=lambda item: (item[0], str(item[1])))[1]
            if fresh_frontiers
            else None
        )
        if (
            selected_render_dir is None
            or render_manifest_path is None
            or frontier_path is None
        ):
            return
        artifacts = [
            MemoryArtifactInput(
                path=Path(os.path.abspath(labels_path)),
                role="candidate_labels",
                media_type="application/octet-stream",
            ),
            MemoryArtifactInput(
                path=Path(os.path.abspath(render_manifest_path)),
                role="selected_only_render_manifest",
                media_type="application/json",
            ),
            MemoryArtifactInput(
                path=Path(os.path.abspath(frontier_path)),
                role="frontier_audit",
                media_type="application/json",
            ),
        ]
        if selected_images:
            artifacts.append(
                MemoryArtifactInput(
                    path=Path(os.path.abspath(selected_images[0])),
                    role="selected_only_review",
                    media_type="image/png",
                )
            )
        try:
            record_observation(
                RememberRequest(
                    workflow="mesh-segmentation",
                    phase="revision_review",
                    interaction=MemoryInteraction(
                        operation="checkpoint_reviewed_candidate",
                        target_object_ids=(part_name,),
                        target_prim_paths=(
                            (target_prim_path,) if target_prim_path else ()
                        ),
                    ),
                    outcome=MemoryOutcome(
                        classification="not_checked",
                        summary=(
                            f"Launcher checkpointed {part_name} "
                            f"{revision_dir.name} after selected-only and "
                            "frontier evidence became available. Semantic "
                            "assessment remains agent-owned and unresolved."
                        ),
                    ),
                    artifacts=tuple(artifacts),
                    importance="high",
                    tags=("mesh-segmentation", part_name, "launcher-checkpoint"),
                )
            )
        except Exception as exc:  # noqa: BLE001 - retry on the next poll
            signature = (part_name, revision_dir.name, type(exc).__name__)
            if signature not in reported_failures:
                reported_failures.add(signature)
                LOGGER.warning(
                    "Could not checkpoint agent memory for %s %s (%s); "
                    "segmentation will continue and the watchdog will retry",
                    part_name,
                    revision_dir.name,
                    type(exc).__name__,
                )
            return
        captured.add(identity)

    def watchdog() -> WatchdogFailure | None:
        for part_name in discover_part_names():
            part_dir = _resolve_part_dir(run_dir, part_name)
            if part_dir is None:
                continue
            revision_dirs = sorted(
                path
                for path in part_dir.glob("rev-*")
                if path.is_dir()
                and path.resolve() == path
                and re.fullmatch(r"rev-\d{3}", path.name)
            )
            for revision_dir in revision_dirs:
                checkpoint_revision(part_name, part_dir, revision_dir)
        return None

    return watchdog


def _write_binary(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Atomically replace one binary artifact with launcher-owned bytes."""

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parent_terminal_promotion_source(
    run_dir: Path,
    *,
    target_parts: list[str],
    include_part_queue: bool = True,
) -> tuple[bytes, Path, str] | None:
    """Return the final ordered lock state after every full lock passes."""

    authoritative_request = {
        "required_skills": ["content-workflow-mesh-segmentation"],
        "inputs": {"target_semantic_parts": list(target_parts)},
    }
    lock_validation = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=(),
        authoritative_request=authoritative_request,
        part_locks_only=True,
        include_part_queue=include_part_queue,
    )
    if not lock_validation["valid"]:
        return None

    lock_digests: set[str] = set()
    for target in target_parts:
        part_dir = _resolve_part_dir(run_dir, target)
        if part_dir is None:
            return None
        validation, validation_error = _load_json_object(
            part_dir / "falsification_validation.json",
            label=f"{target} falsification validation",
        )
        if validation_error or validation is None:
            return None
        candidate_path = _manifest_run_artifact_path(
            run_dir,
            validation.get("candidate_labels"),
        )
        if candidate_path is None or not candidate_path.is_file():
            return None
        candidate_digest = _sha256_file(candidate_path)
        if validation.get("candidate_labels_sha256") != candidate_digest:
            return None
        lock_digests.add(candidate_digest)

    numbered_states: list[tuple[int, Path]] = []
    state_dir = run_dir / "state"
    try:
        state_entries = list(state_dir.iterdir())
    except OSError:
        return None
    for candidate in state_entries:
        match = re.fullmatch(r"labels-(\d+)\.u32le", candidate.name)
        if match is None:
            continue
        try:
            metadata = candidate.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or candidate.resolve() != candidate
            ):
                continue
        except OSError:
            continue
        numbered_states.append((int(match.group(1)), candidate))
    if not numbered_states:
        return None

    numbered_state_digests = {
        _sha256_file(candidate) for _sequence, candidate in numbered_states
    }
    # A targeted agent may lock the requested parts in any evidence-driven
    # order. Bind every final lock to the append-only state chain, then accept
    # the newest state when it is bound to whichever part locked last.
    if not lock_digests.issubset(numbered_state_digests):
        return None
    _sequence, source = max(numbered_states, key=lambda item: item[0])
    source_digest = _sha256_file(source)
    if source_digest not in lock_digests:
        return None
    payload = source.read_bytes()
    if not payload or len(payload) % 4:
        return None
    return payload, source, source_digest


def _regular_single_link_file(path: Path, *, root: Path) -> bool:
    """Return whether ``path`` is one non-linked regular file at that path."""

    try:
        metadata = path.lstat()
        relative = path.relative_to(root)
        expected = root.resolve() / relative
        return (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_nlink == 1
            and path.resolve() == expected
        )
    except (OSError, ValueError):
        return False


def _terminal_deferral_binding(
    run_dir: Path,
    *,
    target: str,
) -> dict[str, object] | None:
    """Bind one explicit child deferral without trusting its diagnosis.

    The child is allowed to declare that it cannot complete a requested target,
    but that declaration is never success evidence.  In particular, none of
    its component counts or candidate-label paths are consumed here.  The
    launcher independently proves the no-unlocked-face condition below.
    """

    part_dir = _resolve_part_dir(run_dir, target)
    if part_dir is None:
        return None
    deferral_path = part_dir / "part_deferral.json"
    if not _regular_single_link_file(deferral_path, root=run_dir):
        return None
    deferral, error = _load_json_object(
        deferral_path,
        label=f"{target} part deferral",
    )
    if error or deferral is None:
        return None
    if any(
        deferral.get(key) != value
        for key, value in {
            "schema_version": "mesh-segmentation-part-deferral.v1",
            "status": "deferred",
            "semantic_part": target,
            "faces_reserved": False,
            "revisit_required": False,
            "revisit_completed": True,
        }.items()
    ):
        return None
    segment_id = deferral.get("segment_id")
    if (
        not isinstance(segment_id, int)
        or isinstance(segment_id, bool)
        or segment_id <= 0
    ):
        return None
    if any(
        not isinstance(deferral.get(key), str) or not deferral[key].strip()
        for key in ("reason", "reason_code")
    ):
        return None
    return {
        "semantic_part": target,
        "segment_id": segment_id,
        "part_deferral": deferral_path.relative_to(run_dir).as_posix(),
        "part_deferral_sha256": _sha256_file(deferral_path),
    }


def _launcher_source_topology(
    staged_asset: Path,
    *,
    target_prim_path: str | None,
) -> tuple[int, str, str] | None:
    """Read source face count and topology directly from the frozen USD."""

    try:
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.Open(str(staged_asset))
        if stage is None:
            return None
        if target_prim_path:
            prim = stage.GetPrimAtPath(target_prim_path)
            if not prim or not prim.IsA(UsdGeom.Mesh):
                return None
            mesh = UsdGeom.Mesh(prim)
        else:
            meshes = [
                UsdGeom.Mesh(prim)
                for prim in stage.Traverse()
                if prim.IsA(UsdGeom.Mesh)
            ]
            if len(meshes) != 1:
                return None
            mesh = meshes[0]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        if not len(counts) or np.any(counts != 3):
            return None
        indices = np.asarray(
            mesh.GetFaceVertexIndicesAttr().Get(),
            dtype=np.int32,
        ).reshape(-1, 3)
        raw_points = mesh.GetPointsAttr().Get()
        if raw_points is None:
            return None
        matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
            mesh.GetPrim()
        )
        points = np.asarray(
            [
                matrix.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
                for point in raw_points
            ],
            dtype=np.float32,
        )
        digest = hashlib.sha256()
        digest.update(points.astype("<f4", copy=False).tobytes())
        digest.update(indices.astype("<i4", copy=False).tobytes())
        return len(counts), f"sha256:{digest.hexdigest()}", str(mesh.GetPath())
    except Exception:  # noqa: BLE001 - any read failure disables early termination
        return None


def _terminal_deferral_failure_source(
    run_dir: Path,
    *,
    target_parts: list[str],
    staged_asset: Path,
    staged_asset_sha256: str,
    target_prim_path: str | None,
    launcher_observed_lock_seals: Mapping[str, Mapping[str, str]],
) -> dict[str, object] | None:
    """Return a launcher-proven terminal-failure receipt, if one is inevitable.

    This is intentionally narrower than terminal validation.  It cannot turn a
    partial result into success and never publishes ``state/final_labels``.  It
    merely stops retrying after the launcher proves that every source face is
    already held by fully validated immutable locks and each remaining target
    was explicitly deferred after its required revisit.
    """

    if not target_parts or not _regular_single_link_file(staged_asset, root=run_dir):
        return None
    if _sha256_file(staged_asset) != staged_asset_sha256:
        return None
    source_topology = _launcher_source_topology(
        staged_asset,
        target_prim_path=target_prim_path,
    )
    if source_topology is None:
        return None
    source_face_count, source_topology_digest, source_target_prim = source_topology

    directories = _part_directories(run_dir)
    locked_targets: list[str] = []
    deferred_bindings: list[dict[str, object]] = []
    for target in target_parts:
        part_dir = directories.get(_normalize_part_key(target))
        completion: dict[str, object] | None = None
        if part_dir is not None:
            completion, completion_error = _load_json_object(
                part_dir / "part_completion.json",
                label=f"{target} part completion",
            )
            if completion_error:
                completion = None
        if completion is not None and completion.get("status") == "locked":
            locked_targets.append(target)
            continue
        binding = _terminal_deferral_binding(run_dir, target=target)
        if binding is None:
            return None
        deferred_bindings.append(binding)
    if not locked_targets or not deferred_bindings:
        return None
    if {_normalize_part_key(target) for target in locked_targets} != set(
        launcher_observed_lock_seals
    ):
        return None

    required = [
        *REQUIRED_FINAL_ARTIFACTS,
        *REQUIRED_TARGETED_FINAL_ARTIFACTS,
    ]
    missing = [relative for relative in required if not (run_dir / relative).is_file()]
    # Partial final labels remain unsafe to publish.  Every other requested
    # artifact must already exist so this path cannot truncate useful work.
    if missing != ["state/final_labels.u32le"]:
        return None

    promotion = _parent_terminal_promotion_source(
        run_dir,
        target_parts=locked_targets,
        # The authoritative target subset is deliberate here.  The ordinary
        # terminal validator still includes every queue entry and must fail the
        # whole run because the deferred targets are not locked.
        include_part_queue=False,
    )
    if promotion is None:
        return None
    payload, latest_state, latest_state_sha256 = promotion

    locked_bindings: list[dict[str, object]] = []
    locked_segment_ids: set[int] = set()
    for target in locked_targets:
        part_dir = _resolve_part_dir(run_dir, target)
        if part_dir is None:
            return None
        completion_path = part_dir / "part_completion.json"
        decision_path = part_dir / "initializer_decision.json"
        gate_path = part_dir / "initializer_runtime_gate.json"
        if not all(
            _regular_single_link_file(path, root=run_dir)
            for path in (completion_path, decision_path, gate_path)
        ):
            return None
        completion, completion_error = _load_json_object(
            completion_path,
            label=f"{target} part completion",
        )
        decision, decision_error = _load_json_object(
            decision_path,
            label=f"{target} initializer decision",
        )
        gate, gate_error = _load_json_object(
            gate_path,
            label=f"{target} initializer runtime gate",
        )
        if (
            completion_error
            or decision_error
            or gate_error
            or completion is None
            or decision is None
            or gate is None
        ):
            return None
        observed_seals = dict(
            launcher_observed_lock_seals.get(_normalize_part_key(target), {})
        )
        if not observed_seals or gate.get("sealed_digests") != observed_seals:
            return None
        for relative, observed_digest in observed_seals.items():
            sealed_path = _safe_run_artifact_path(run_dir, relative)
            if (
                not re.fullmatch(r"[0-9a-f]{64}", observed_digest)
                or sealed_path is None
                or not _regular_single_link_file(sealed_path, root=run_dir)
                or _sha256_file(sealed_path) != observed_digest
            ):
                return None
        segment_id = completion.get("segment_id")
        if (
            not isinstance(segment_id, int)
            or isinstance(segment_id, bool)
            or segment_id <= 0
            or decision.get("segment_id") != segment_id
            or segment_id in locked_segment_ids
            or gate.get("schema_version") != INITIALIZER_RUNTIME_GATE_SCHEMA_VERSION
            or gate.get("status") != "accepted"
            or gate.get("locked_observed") is not True
        ):
            return None
        locked_segment_ids.add(segment_id)
        locked_bindings.append(
            {
                "semantic_part": target,
                "segment_id": segment_id,
                "part_completion": completion_path.relative_to(run_dir).as_posix(),
                "part_completion_sha256": _sha256_file(completion_path),
                "initializer_runtime_gate": gate_path.relative_to(run_dir).as_posix(),
                "initializer_runtime_gate_sha256": _sha256_file(gate_path),
                "launcher_observed_initializer_sealed_digests": observed_seals,
            }
        )

    deferred_segment_ids = {int(binding["segment_id"]) for binding in deferred_bindings}
    if (
        len(deferred_segment_ids) != len(deferred_bindings)
        or deferred_segment_ids & locked_segment_ids
    ):
        return None

    topology_path = run_dir / "prepare" / "topology.json"
    fragment_manifest_path = run_dir / "fragments" / "fragment_manifest.json"
    fragment_ids_path = run_dir / "fragments" / "fragment_ids.u32le"
    if not all(
        _regular_single_link_file(path, root=run_dir)
        for path in (topology_path, fragment_manifest_path, fragment_ids_path)
    ):
        return None
    topology, topology_error = _load_json_object(
        topology_path,
        label="prepare/topology.json",
    )
    fragment_manifest, fragment_manifest_error = _load_json_object(
        fragment_manifest_path,
        label="fragments/fragment_manifest.json",
    )
    if (
        topology_error
        or fragment_manifest_error
        or topology is None
        or fragment_manifest is None
    ):
        return None
    face_count = topology.get("source_face_count")
    topology_digest = topology.get("topology_digest")
    fragment_manifest_ids = _manifest_run_artifact_path(
        run_dir,
        fragment_manifest.get("fragment_ids"),
    )
    if (
        not isinstance(face_count, int)
        or isinstance(face_count, bool)
        or face_count <= 0
        or fragment_manifest.get("source_face_count") != face_count
        or face_count != source_face_count
        or fragment_manifest_ids != fragment_ids_path
        or fragment_manifest.get("fragment_ids_sha256")
        != _sha256_file(fragment_ids_path)
        or topology.get("source_sha256") != staged_asset_sha256
        or fragment_manifest.get("source_sha256") != staged_asset_sha256
        or fragment_manifest.get("topology_digest") != topology_digest
        or topology_digest != source_topology_digest
        or topology.get("target_prim_path") != source_target_prim
        or fragment_manifest.get("target_prim_path") != source_target_prim
        or not isinstance(topology_digest, str)
        or not topology_digest.startswith("sha256:")
        or fragment_ids_path.stat().st_size != face_count * 4
        or len(payload) != face_count * 4
    ):
        return None

    latest_labels, latest_labels_error = _load_u32_labels(latest_state)
    if (
        not _regular_single_link_file(latest_state, root=run_dir)
        or latest_labels_error
        or latest_labels is None
    ):
        return None
    labels_present = set(latest_labels)
    # This is the decisive launcher-owned proof.  No residual, unassigned, or
    # deferred-target face exists.  Creating a nonempty deferred target would
    # therefore require changing a fully validated immutable lock.
    if (
        len(latest_labels) != face_count
        or labels_present != locked_segment_ids
        or 0 in labels_present
    ):
        return None

    return {
        "schema_version": TERMINAL_DEFERRAL_FAILURE_SCHEMA_VERSION,
        "status": "failed",
        "receipt_owner": "launcher",
        "failure_code": "requested_targets_deferred_no_unlocked_faces",
        "checked_at": utc_now(),
        "requested_target_parts": list(target_parts),
        "locked_targets": locked_bindings,
        "deferred_targets": deferred_bindings,
        "face_universe": {
            "source_asset": staged_asset.relative_to(run_dir).as_posix(),
            "source_asset_sha256": staged_asset_sha256,
            "source_face_count": face_count,
            "topology": topology_path.relative_to(run_dir).as_posix(),
            "topology_sha256": _sha256_file(topology_path),
            "topology_digest": topology_digest,
            "target_prim_path": source_target_prim,
            "fragment_ids": fragment_ids_path.relative_to(run_dir).as_posix(),
            "fragment_ids_sha256": _sha256_file(fragment_ids_path),
            "latest_locked_state": latest_state.relative_to(run_dir).as_posix(),
            "latest_locked_state_sha256": latest_state_sha256,
            "locked_segment_ids": sorted(locked_segment_ids),
            "unlocked_face_count": 0,
            "residual_face_count": 0,
        },
        "publication": {
            "state/final_labels.u32le": "withheld",
            "partial_labels_published": False,
        },
        "terminal_contract": {
            "terminal_validation_waived": False,
            "run_success": False,
        },
    }


def _remove_child_terminal_artifact(path: Path) -> None:
    """Discard any child-authored terminal path before parent reconciliation."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode)):
        raise ValueError(f"Child terminal artifact has unsafe type: {path}")
    path.unlink()


def _remove_child_terminal_labels(path: Path) -> None:
    """Compatibility wrapper for the terminal-label reconciliation path."""

    _remove_child_terminal_artifact(path)


def _publish_parent_terminal_labels(
    *,
    run_dir: Path,
    payload: bytes,
    source: Path,
    digest: str,
    observed_at: str,
) -> None:
    final_labels_path = run_dir / "state" / "final_labels.u32le"
    _write_binary(final_labels_path, payload)
    _write_json(
        run_dir / "raw" / "final_labels_promotion.json",
        {
            "schema_version": FINAL_LABELS_PROMOTION_SCHEMA_VERSION,
            "promotion_owner": "launcher",
            "observed_at": observed_at,
            "final_labels": str(final_labels_path),
            "final_labels_sha256": digest,
            "source_labels": str(source.relative_to(run_dir)),
            "source_labels_sha256": digest,
        },
    )


def _run_isolated_child_agent(
    *,
    config: MeshSegmentationConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    prompt_image_inputs: list[dict[str, str]],
    bridge_artifact_prefix: str,
    output_schema: dict[str, object] | None = None,
    enable_workflow_watchdog: bool = True,
    agent_memory: AgentMemory | None = None,
    agent_memory_remember: Callable[[RememberRequest], object] | None = None,
    trusted_script_digests: dict[str, str] | None = None,
    progress_tracker: MeshSegmentationProgressTracker | None = None,
) -> int:
    # Freeze launcher-selected helper trust once for the whole continuation
    # loop. Later turns must never bless bytes left behind by an earlier child.
    trusted_script_digests = dict(trusted_script_digests or {})
    final_labels_path = run_dir / "state" / "final_labels.u32le"
    terminal_deferral_failure_path = run_dir / "raw" / "terminal_deferral_failure.json"
    target_parts = list(config.target_semantic_parts or [])
    parent_promotion: tuple[bytes, Path, str] | None = None
    parent_promotion_observed_at: str | None = None
    parent_terminal_deferral_failure: dict[str, object] | None = None
    staged_asset_sha256 = (
        _sha256_file(config.asset_path)
        if _regular_single_link_file(config.asset_path, root=run_dir)
        else None
    )

    def reconcile_parent_terminal_promotion() -> None:
        nonlocal parent_promotion
        nonlocal parent_promotion_observed_at
        nonlocal parent_terminal_deferral_failure
        if not target_parts:
            return
        # A child-created path is never promotion evidence. Remove it after the
        # turn, then publish only bytes selected by the launcher's full lock
        # validator. Once promoted, both time and bytes remain frozen.
        _remove_child_terminal_labels(final_labels_path)
        _remove_child_terminal_artifact(terminal_deferral_failure_path)
        if parent_promotion is None:
            parent_promotion = _parent_terminal_promotion_source(
                run_dir,
                target_parts=target_parts,
            )
            if parent_promotion is not None:
                parent_promotion_observed_at = datetime.now(UTC).isoformat()
        if parent_promotion is not None:
            assert parent_promotion_observed_at is not None
            payload, source, digest = parent_promotion
            _publish_parent_terminal_labels(
                run_dir=run_dir,
                payload=payload,
                source=source,
                digest=digest,
                observed_at=parent_promotion_observed_at,
            )
            return
        if parent_terminal_deferral_failure is None and staged_asset_sha256 is not None:
            parent_terminal_deferral_failure = _terminal_deferral_failure_source(
                run_dir,
                target_parts=target_parts,
                staged_asset=config.asset_path,
                staged_asset_sha256=staged_asset_sha256,
                target_prim_path=config.target_prim_path,
                launcher_observed_lock_seals=_watchdog_observed_lock_seals(
                    initializer_watchdog
                ),
            )
        if parent_terminal_deferral_failure is not None:
            # This receipt is failure provenance, never a success artifact. It
            # is frozen in launcher memory and republished after the child has
            # exited, just like successful final-label promotion.
            _write_json(
                terminal_deferral_failure_path,
                parent_terminal_deferral_failure,
            )

    if not enable_workflow_watchdog:
        initializer_watchdog = None
        review_watchdog = None
        memory_checkpoint_watchdog = None
        workflow_watchdog = progress_tracker.poll if progress_tracker else None
    else:
        initializer_watchdog = _make_initialization_watchdog(
            run_dir,
            expected_parts=list(config.target_semantic_parts or []),
        )
        if config.target_semantic_parts:
            # Targeted runs have no launcher-side review gate: the independent
            # visual reviewer is retired. Selection evidence is still validated
            # from the run's own artifacts at terminal validation.
            review_watchdog = None
        else:
            # Recognition keeps its structural sequential-lock gate, which
            # enforces queue and lock ordering without any vision review.
            review_watchdog = _make_recognition_sequential_watchdog(run_dir)
        memory_checkpoint_watchdog = _make_agent_memory_checkpoint_watchdog(
            run_dir,
            target_parts=config.target_semantic_parts,
            memory=agent_memory,
            remember=agent_memory_remember,
            target_prim_path=config.target_prim_path,
        )
        workflow_watchdog = _combine_watchdogs(
            initializer_watchdog,
            review_watchdog,
            memory_checkpoint_watchdog,
            progress_tracker.poll if progress_tracker else None,
        )
    # `iteration_budget` has always been published to the child as a constraint
    # but never enforced: the launcher ran exactly one turn. A model that ended
    # its turn with work outstanding therefore lost the whole case, even with
    # most of its time budget unspent. Give it the follow-up turns the setting
    # already promises. Each turn re-reads the run directory, so a continuation
    # resumes from the artifacts already on disk.
    previous_counts: tuple[int, int, int] | None = None
    idle_turns = 0
    usage_totals: dict[str, int] = {}
    case_started = time.monotonic()
    case_budget = max(0.0, config.case_timeout_seconds)
    # The budget check below can break before the first turn runs, which left
    # `returncode` unbound and raised `UnboundLocalError` instead of reporting
    # the exhausted case. A case that never got a turn did not succeed.
    returncode = 1
    for attempt in range(1, max(1, config.iteration_budget) + 1):
        elapsed = time.monotonic() - case_started
        if case_budget and elapsed >= case_budget:
            # Stop granting turns rather than starting one that would run past
            # the budget. Reported explicitly so an exhausted case is
            # distinguishable from one that finished or gave up.
            print(
                "content-workflow-cli: case wall-clock budget exhausted after "
                f"{elapsed:.0f}s of {case_budget:.0f}s ({attempt - 1} turns); "
                "stopping",
                flush=True,
            )
            break
        # Clamp this turn to what is left of the case budget. Checking only
        # before starting a turn bounded the case at case_timeout +
        # child_timeout, not case_timeout: a turn beginning just under the
        # budget still ran a full per-turn timeout past it.
        turn_config = config
        if case_budget:
            remaining = case_budget - elapsed
            if (
                config.child_timeout_seconds <= 0
                or remaining < config.child_timeout_seconds
            ):
                turn_config = replace(config, child_timeout_seconds=remaining)
        turn_prompt = (
            prompt
            if attempt == 1
            else _continuation_prompt(prompt, run_dir, attempt=attempt, config=config)
        )
        outstanding_before = _outstanding_work(run_dir, config=config)
        locked_before = _outstanding_progress_units(
            outstanding_before,
            target_count=len(target_parts),
        )
        timeout_error: TimeoutError | None = None
        try:
            if config.codex_execution_mode == CODEX_EXECUTION_HOST:
                returncode = _run_child_agent(
                    config=turn_config,
                    prompt=turn_prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    scene_service=None,
                    prompt_image_inputs=prompt_image_inputs,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                    output_schema=output_schema,
                    additional_watchdog=workflow_watchdog,
                    trusted_script_digests=trusted_script_digests,
                )
            else:
                returncode = _run_child_agent_codex_container(
                    config=turn_config,
                    prompt=turn_prompt,
                    run_dir=run_dir,
                    child_output_path=child_output_path,
                    child_final_path=child_final_path,
                    prompt_image_inputs=prompt_image_inputs,
                    bridge_artifact_prefix=bridge_artifact_prefix,
                    output_schema=output_schema,
                    workflow_watchdog=workflow_watchdog,
                    trusted_script_digests=trusted_script_digests,
                )
        except TimeoutError as exc:
            timeout_error = exc
            returncode = 0
        if progress_tracker is not None:
            progress_tracker.poll()
        reconcile_parent_terminal_promotion()
        _record_turn_usage(
            run_dir,
            bridge_artifact_prefix=bridge_artifact_prefix,
            attempt=attempt,
            totals=usage_totals,
        )
        outstanding = _outstanding_work(run_dir, config=config)
        locked_after = _outstanding_progress_units(
            outstanding,
            target_count=len(target_parts),
        )
        semantic_progress_detected = _outstanding_made_durable_progress(
            outstanding_before,
            outstanding,
        )
        if timeout_error is not None:
            _record_child_turn_timeout(
                run_dir,
                attempt=attempt,
                timeout_seconds=turn_config.child_timeout_seconds,
                locked_before=locked_before,
                locked_after=locked_after,
                semantic_progress_detected=semantic_progress_detected,
                progress_stage=(
                    progress_tracker.last_stage
                    if progress_tracker is not None
                    else None
                ),
                progress_summary=(
                    progress_tracker.last_summary
                    if progress_tracker is not None
                    else None
                ),
            )
            if not semantic_progress_detected:
                raise timeout_error
            print(
                "content-workflow-cli: child turn timed out after durable semantic "
                "progress; preserving evidence",
                flush=True,
            )
        if returncode != 0:
            break
        if parent_terminal_deferral_failure is not None:
            deferred = ", ".join(
                str(binding["semantic_part"])
                for binding in parent_terminal_deferral_failure["deferred_targets"]
            )
            print(
                "content-workflow-cli: terminal target deferral verified for "
                f"{deferred}; final labels remain unpublished and the run will "
                "fail terminal validation; stopping continuations",
                flush=True,
            )
            break
        if not outstanding:
            break
        counts = outstanding.counts
        if counts == previous_counts:
            idle_turns += 1
        else:
            idle_turns = 0
        previous_counts = counts
        # A turn with no measurable change is not evidence of being stuck. One
        # observed run reported the identical outstanding set for four
        # consecutive turns -- because this agent does its locking and
        # bookkeeping at the very end -- and then completed every part on the
        # fifth. Stopping at the first idle turn would have failed a run that
        # succeeded. Only give up after sustained idleness.
        if idle_turns >= _MAX_IDLE_CONTINUATION_TURNS:
            print(
                "content-workflow-cli: "
                f"{idle_turns} continuation turns made no progress "
                f"({outstanding.description}); stopping early",
                flush=True,
            )
            break
        if attempt == max(1, config.iteration_budget):
            print(
                "content-workflow-cli: iteration budget exhausted with work "
                f"outstanding: {outstanding.description}",
                flush=True,
            )
            break
        print(
            f"content-workflow-cli: child ended with work outstanding "
            f"({outstanding.description}); starting continuation turn "
            f"{attempt + 1}/{config.iteration_budget}",
            flush=True,
        )
    if returncode == 0 and initializer_watchdog is not None:
        # A child may publish rev-000 immediately before exiting. Poll twice so
        # the launcher first observes a stable identity and then validates it.
        for _ in range(2):
            initialization_failure = initializer_watchdog()
            if initialization_failure is not None:
                return 1
    if memory_checkpoint_watchdog is not None:
        memory_checkpoint_watchdog()
    if progress_tracker is not None:
        progress_tracker.poll()
    reconcile_parent_terminal_promotion()
    return returncode


def _mesh_segmentation_helper_scripts(
    config: MeshSegmentationConfig,
) -> list[Path]:
    """Return checked-in fragment and selected-workflow evidence tools.

    The child sandbox can read these repository-owned files but cannot rewrite
    them. Copying the same tools into the writable run directory would turn the
    pre-execution digest check into a time-of-check/time-of-use boundary.
    """

    canonical_candidates = (
        config.repo_root
        / "agentic/.agents/skills/content-workflow-mesh-segmentation/scripts",
        Path(__file__).resolve().parents[4]
        / "agentic/.agents/skills/content-workflow-mesh-segmentation/scripts",
    )
    canonical_root = next(
        (
            path
            for path in canonical_candidates
            if (path / "prepare_mesh.py").is_file()
            and (path / "oversegment_mesh.py").is_file()
        ),
        canonical_candidates[0],
    )
    if not canonical_root.is_dir():
        raise FileNotFoundError(
            f"Mesh-segmentation skill scripts are missing: {canonical_root}"
        )
    script_roots = [canonical_root]
    if config.workflow_skill != "content-workflow-mesh-segmentation":
        workflow_candidates = (
            config.repo_root
            / "agentic/.agents/skills"
            / config.workflow_skill
            / "scripts",
            Path(__file__).resolve().parents[4]
            / "agentic/.agents/skills"
            / config.workflow_skill
            / "scripts",
        )
        workflow_root = next(
            (
                path
                for path in workflow_candidates
                if (path / "render_face_id_buffers.py").is_file()
            ),
            workflow_candidates[0],
        )
        if not workflow_root.is_dir():
            raise FileNotFoundError(
                f"Selected workflow scripts are missing: {workflow_root}"
            )
        script_roots.append(workflow_root)

    helpers: list[Path] = []
    helper_names: dict[str, Path] = {}
    for script_root in script_roots:
        for source in sorted(
            path
            for path in script_root.iterdir()
            if path.is_file() and path.suffix in {".py", ".cpp"}
        ):
            prior = helper_names.get(source.name)
            if prior is not None:
                raise ValueError(
                    "Mesh-segmentation helper script name collision: "
                    f"{prior} and {source}"
                )
            helper_names[source.name] = source
            helpers.append(source.resolve())
    return helpers


def _run_child_agent_codex_container(
    *,
    config: MeshSegmentationConfig,
    prompt: str,
    run_dir: Path,
    child_output_path: Path,
    child_final_path: Path,
    prompt_image_inputs: list[dict[str, str]],
    bridge_artifact_prefix: str,
    output_schema: dict[str, object] | None = None,
    workflow_watchdog: Callable[[], WatchdogFailure | None] | None = None,
    trusted_script_digests: dict[str, str] | None = None,
) -> int:
    """Run a fresh Codex turn without mounting the repository or sibling runs."""

    _reject_unsafe_run_links(run_dir)
    _stage_agent_skills(config, run_dir)

    package_root = Path(__file__).resolve().parents[1]
    bridge_path = package_root / "content_workflow_cli" / "codex_sdk_bridge.mjs"
    node_modules = package_root / "node_modules"
    auth_path = _effective_codex_home(child_cwd=run_dir) / "auth.json"
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    for required in (bridge_path, node_modules, auth_path, purelib):
        if not required.exists():
            raise FileNotFoundError(
                f"Containerized Codex runtime dependency is missing: {required}"
            )
    _ensure_container_codex_runtime_permissions(node_modules)
    _stage_container_python_runtime(
        run_dir=run_dir,
        purelib=purelib,
    )
    container_environment = _codex_bridge_env(config, run_dir)

    artifact_prefix = bridge_artifact_prefix
    sdk_request_path = run_dir / "raw" / f"{artifact_prefix}_request.json"
    sdk_request = _build_codex_sdk_request(
        config=config,
        prompt=prompt,
        run_dir=run_dir,
        child_final_path=child_final_path,
        prompt_image_inputs=prompt_image_inputs,
        output_schema=output_schema,
        bridge_artifact_prefix=artifact_prefix,
        trusted_script_digests=trusted_script_digests,
    )
    _write_private_json(sdk_request_path, sdk_request)

    host_uid = os.getuid()
    host_gid = os.getgid()
    container_name = _container_name_for_run(run_dir)
    command = _build_codex_container_command(
        config=config,
        run_dir=run_dir,
        sdk_request_path=sdk_request_path,
        bridge_path=bridge_path,
        node_modules=node_modules,
        auth_path=auth_path,
        container_name=container_name,
        host_uid=host_uid,
        host_gid=host_gid,
        container_environment=container_environment,
        trusted_script_digests=trusted_script_digests,
    )

    try:
        with child_output_path.open("w", encoding="utf-8") as log_stream:
            log_stream.write(
                "$ docker run <isolated mesh-segmentation child container>\n"
            )
            log_stream.write(f"image: {config.codex_container_image}\n")
            log_stream.write(f"request: {sdk_request_path}\n")
            log_stream.write(
                "mount policy: current run=rw; runtime/auth dependencies=ro; "
                "repository and sibling runs=unmounted\n"
            )
            log_stream.flush()
            return _run_subprocess_with_timeout(
                command=command,
                cwd=run_dir,
                env={**os.environ, **container_environment},
                timeout_seconds=config.child_timeout_seconds,
                log_stream=log_stream,
                timeout_label="containerized codex mesh-segmentation child turn",
                scene_watchdog=workflow_watchdog,
                progress_reporter=_run_progress_summary_for_log,
                run_dir=run_dir,
                terminal_success_detector=_terminal_success_detector_for_bridge(
                    artifact_prefix
                ),
                console_stream=_console_stream(config),
            )
    finally:
        _remove_named_container(container_name)
        _reject_unsafe_run_links(run_dir)


def _build_codex_container_command(
    *,
    config: MeshSegmentationConfig,
    run_dir: Path,
    sdk_request_path: Path,
    bridge_path: Path,
    node_modules: Path,
    auth_path: Path,
    container_name: str,
    host_uid: int,
    host_gid: int,
    container_environment: Mapping[str, str],
    trusted_script_digests: dict[str, str] | None = None,
) -> list[str]:
    container_bridge = "/opt/content-workflow-cli/codex_sdk_bridge.mjs"
    pythonpath = str(run_dir / ".runtime" / "site-packages")
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{host_uid}:{host_gid}",
        "--network",
        "host",
        "--env",
        "HOME=/tmp/codex-home",
        "--env",
        f"PYTHONPATH={pythonpath}",
    ]
    for variable in sorted(container_environment):
        if _forward_container_environment_variable(variable):
            command.extend(["--env", variable])
    command.extend(
        [
            "--mount",
            _docker_bind_mount(
                Path("/usr/bin/node"), Path("/usr/bin/node"), readonly=True
            ),
            "--mount",
            _docker_bind_mount(bridge_path, Path(container_bridge), readonly=True),
            "--mount",
            _docker_bind_mount(
                node_modules,
                Path("/opt/content-workflow-cli/node_modules"),
                readonly=True,
            ),
            "--mount",
            _docker_bind_mount(
                auth_path,
                Path("/tmp/codex-home/.codex/auth.json"),
                readonly=True,
            ),
        ]
    )
    trusted_roots: set[Path] = set()
    for raw_path in trusted_script_digests or {}:
        script_path = Path(raw_path)
        if not script_path.is_absolute():
            raise ValueError(f"Trusted script path must be absolute: {script_path}")
        try:
            metadata = script_path.lstat()
        except OSError as exc:
            raise ValueError(f"Trusted script is unavailable: {script_path}") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or script_path.resolve() != script_path
        ):
            raise ValueError(f"Trusted script has an unsafe file type: {script_path}")
        try:
            script_path.relative_to(run_dir.resolve())
        except ValueError:
            trusted_roots.add(script_path.parent)
        else:
            raise ValueError(
                f"Trusted script cannot be inside the writable run: {script_path}"
            )
    for trusted_root in sorted(trusted_roots):
        command.extend(
            [
                "--mount",
                _docker_bind_mount(trusted_root, trusted_root, readonly=True),
            ]
        )
    command.extend(
        [
            "--mount",
            _docker_bind_mount(run_dir, run_dir, readonly=False),
            "--mount",
            _docker_bind_mount(
                run_dir / ".runtime",
                run_dir / ".runtime",
                readonly=True,
            ),
            "--entrypoint",
            "/usr/bin/node",
            config.codex_container_image,
            container_bridge,
            str(sdk_request_path),
        ]
    )
    return command


def _forward_container_environment_variable(name: str) -> bool:
    """Forward only launcher-owned child capabilities, never their values."""

    return (
        name
        in {
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "TRACEPARENT",
            "TRACESTATE",
            "USD_CLI_AGENT",
            "USD_CLI_ATTACHED_PROJECT_DIR",
            "USD_CLI_LIFECYCLE_EXTERNALLY_OWNED",
            "USD_CLI_LOCAL_GPU_FORBIDDEN",
            "USD_CLI_NO_DAEMON",
            "WARP_CACHE_PATH",
            "XDG_CACHE_HOME",
        }
        or name.startswith("CONTENT_WORKFLOW_PARENT_USD_CLI_")
        or name.startswith("CONTENT_WORKFLOW_USD_CLI_")
        or name.startswith("USD_CLI_TEL_")
    )


def _stage_container_python_runtime(
    *,
    run_dir: Path,
    purelib: Path,
) -> None:
    """Stage the small pure-Python runtime needed inside the nested sandbox."""

    destination = run_dir / ".runtime" / "site-packages"
    if destination.parent.exists():
        shutil.rmtree(destination.parent)
    destination.mkdir(parents=True, exist_ok=False)
    packages = (
        (purelib / "trimesh", destination / "trimesh"),
        (purelib / "networkx", destination / "networkx"),
    )
    for source, target in packages:
        if not source.is_dir():
            raise FileNotFoundError(
                f"Containerized mesh-segmentation Python dependency is missing: {source}"
            )
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _docker_bind_mount(source: Path, destination: Path, *, readonly: bool) -> str:
    for value in (str(source), str(destination)):
        if "," in value:
            raise ValueError(f"Docker bind-mount paths cannot contain commas: {value}")
    pieces = ["type=bind", f"src={source}", f"dst={destination}"]
    if readonly:
        pieces.append("readonly")
    return ",".join(pieces)


def _container_name_for_run(run_dir: Path) -> str:
    digest = hashlib.sha256(str(run_dir).encode("utf-8")).hexdigest()[:16]
    return f"content-mesh-segmentation-{digest}"


def _ensure_container_codex_runtime_permissions(node_modules: Path) -> None:
    """Let the inner unprivileged sandbox traverse only the Codex native runtime."""

    openai_root = node_modules / "@openai"
    native_packages = sorted(openai_root.glob("codex-linux-*"))
    if not native_packages:
        raise FileNotFoundError(
            f"Codex native Node runtime is missing under: {openai_root}"
        )
    for root in (node_modules, openai_root, openai_root / "codex", *native_packages):
        if not root.exists():
            raise FileNotFoundError(f"Codex Node runtime is missing: {root}")
        if root in {node_modules, openai_root}:
            root.chmod(
                root.stat().st_mode
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
            continue
        for current_root, directories, files in os.walk(root):
            current_path = Path(current_root)
            current_path.chmod(
                current_path.stat().st_mode
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
            for directory in directories:
                path = current_path / directory
                path.chmod(
                    path.stat().st_mode
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH
                )
            for filename in files:
                path = current_path / filename
                mode = path.stat().st_mode | stat.S_IRGRP | stat.S_IROTH
                path.chmod(mode)


def _remove_named_container(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "--force", container_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_mesh_segmentation(
    config: MeshSegmentationConfig,
) -> MeshSegmentationResult:
    """Launch one independent child-agent mesh-segmentation run."""

    _validate_config(config)
    run_started = time.monotonic()
    run_id, run_dir = _resolve_run_dir(config)
    memory_root: Path | None = None
    memory: AgentMemory | None = None
    if config.memory_enabled:
        memory_root = resolve_memory_root(
            config.memory_root,
            repo_root=config.repo_root,
            create=False,
        )
        memory_run_dir = memory_root / run_id
        if memory_run_dir == run_dir or memory_run_dir.is_relative_to(run_dir):
            raise ValueError(
                "Mesh-segmentation memory must be outside the child-writable "
                "output directory. Choose a sibling --memory-root."
            )
        if memory_run_dir.exists():
            raise FileExistsError(
                "A fresh mesh-segmentation run requires a new memory run: "
                f"{memory_run_dir}"
            )
    _initialize_run_dir(run_dir)
    if config.memory_enabled:
        assert memory_root is not None
        memory = AgentMemory(
            run_id=run_id,
            memory_root=memory_root,
        )
        memory_root = memory.root
    staged_asset, staged_references, staging = _stage_inputs(config, run_dir)
    if config.continue_from_run is not None:
        staging["continuation"] = _stage_continuation_seed(
            source_run=config.continue_from_run,
            run_dir=run_dir,
            staged_asset=staged_asset,
        )
    runtime_config = replace(
        config,
        asset_path=staged_asset,
        reference_images=staged_references,
        agent_cwd=run_dir,
        agent_workspace=config.repo_root / "agentic",
        usd_cli_session_id=_workflow_usd_cli_session_id(
            workflow="mesh-segmentation",
            project_dir=run_dir,
        ),
    )

    request_path = run_dir / "request.json"
    prompt_path = run_dir / "agent_prompt.md"
    child_output_path = run_dir / "child-output.log"
    child_final_path = run_dir / "child-final.md"
    terminal_validation_path = run_dir / "terminal_validation.json"
    request = _build_request(
        runtime_config,
        run_id=run_id,
        run_dir=run_dir,
        staging=staging,
        memory_root=memory_root,
        memory_broker_url=None,
    )
    _write_json(request_path, request)
    trusted_helper_scripts = _mesh_segmentation_helper_scripts(runtime_config)
    helper_scripts_by_name = {path.name: path for path in trusted_helper_scripts}
    trusted_helper_digests = {
        str(path): _sha256_file(path) for path in trusted_helper_scripts
    }
    prompt = _build_prompt(
        runtime_config,
        request_path=request_path,
        run_dir=run_dir,
        run_id=run_id,
        memory_root=memory_root,
        memory_broker_url=None,
        helper_scripts=helper_scripts_by_name,
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    _chmod_private(prompt_path)

    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "run_created",
        phase="setup",
        summary="Created an isolated mesh-segmentation run and staged its evidence.",
        artifacts=[
            str(request_path),
            str(prompt_path),
            str(staged_asset),
            *[str(path) for path in staged_references],
            *[str(path) for path in trusted_helper_scripts],
            *(
                [str(run_dir / "continuation_seed" / "manifest.json")]
                if config.continue_from_run is not None
                else []
            ),
        ],
        data={
            "workflow": request["workflow"],
            "fresh_child_thread": True,
            "prior_run_access_allowed": False,
        },
    )

    if config.dry_run:
        _stage_agent_skills(runtime_config, run_dir)
        trace_writer.write(
            "dry_run_completed",
            phase="setup",
            summary="Prepared the fresh-session request without launching an agent.",
            artifacts=[str(run_dir / ".agents" / "skills")],
        )
        return MeshSegmentationResult(
            run_dir=run_dir,
            request_path=request_path,
            prompt_path=prompt_path,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            terminal_validation_path=None,
            returncode=0,
            completed=False,
        )

    memory_broker: AgentMemoryBroker | None = None
    parent_usd_cli_capability = None
    progress_tracker: MeshSegmentationProgressTracker | None = None
    child_returncode = 2
    try:
        parent_usd_cli_capability = start_parent_usd_cli_capability(
            config=runtime_config,
            run_dir=run_dir,
            workflow="mesh-segmentation.run",
            session_workflow="mesh-segmentation",
            initial_scene=staged_asset,
            timeout_seconds=max(config.scene_tool_timeout_seconds, 900.0),
        )
        runtime_config = replace(
            runtime_config,
            usd_cli_session_id=parent_usd_cli_capability.session.session_id,
            parent_usd_cli_session_identity=(parent_usd_cli_capability.identity_path),
            parent_usd_cli_session_identity_sha256=(
                parent_usd_cli_capability.identity_sha256
            ),
        )
        prompt += parent_usd_cli_prompt_contract(parent_usd_cli_capability)
        prompt_path.write_text(prompt, encoding="utf-8")
        _chmod_private(prompt_path)
        trace_writer.write(
            "usd_cli_ready",
            phase="setup",
            summary="Opened one workflow-owned usd-cli session with OVRTX ready.",
            artifacts=(
                [str(parent_usd_cli_capability.readiness.artifact_path)]
                if parent_usd_cli_capability.readiness.artifact_path is not None
                else []
            ),
            data={"usd_cli_session_id": parent_usd_cli_capability.session.session_id},
        )
        if memory is not None:
            memory_broker = AgentMemoryBroker(
                run_dir=run_dir,
                memory=memory,
                search_ready_path=run_dir / "part_plan.json",
            )
            memory_broker.start()
            request = _build_request(
                runtime_config,
                run_id=run_id,
                run_dir=run_dir,
                staging=staging,
                memory_root=memory_root,
                memory_broker_url=memory_broker.url,
            )
            _write_json(request_path, request)
            prompt = _build_prompt(
                runtime_config,
                request_path=request_path,
                run_dir=run_dir,
                run_id=run_id,
                memory_root=memory_root,
                memory_broker_url=memory_broker.url,
                helper_scripts=helper_scripts_by_name,
            )
            prompt += parent_usd_cli_prompt_contract(parent_usd_cli_capability)
            prompt_path.write_text(prompt, encoding="utf-8")
            _chmod_private(prompt_path)
            trace_writer.write(
                "agent_memory_broker_started",
                phase="setup",
                summary="Started the wrapper-owned observation-memory broker.",
                data={"access": "launcher_broker"},
            )

        progress_tracker = MeshSegmentationProgressTracker(run_dir, trace_writer)
        progress_tracker.start()
        child_returncode = _run_isolated_child_agent(
            config=runtime_config,
            prompt=prompt,
            run_dir=run_dir,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            prompt_image_inputs=[],
            bridge_artifact_prefix="mesh_segmentation",
            agent_memory=memory,
            agent_memory_remember=(
                memory_broker.remember if memory_broker is not None else None
            ),
            trusted_script_digests=trusted_helper_digests,
            progress_tracker=progress_tracker,
        )
        progress_tracker.poll()
        trace_writer.write(
            "child_agent_finished",
            phase="runner",
            summary="Fresh mesh-segmentation child agent exited.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={
                "returncode": child_returncode,
                "last_observed_stage": progress_tracker.last_stage,
                "last_observed_progress": progress_tracker.last_summary,
            },
        )
    except UnsafeRunArtifactError:
        raise
    except (KeyboardInterrupt, SystemExit):
        if progress_tracker is not None:
            progress_tracker.poll()
        trace_writer.write(
            "child_agent_cancelled",
            phase="runner",
            summary="Mesh-segmentation child-agent runner was cancelled.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={
                "last_observed_stage": (
                    progress_tracker.last_stage
                    if progress_tracker is not None
                    else None
                ),
                "last_observed_progress": (
                    progress_tracker.last_summary
                    if progress_tracker is not None
                    else None
                ),
            },
        )
        raise
    except Exception as exc:  # noqa: BLE001 - preserve partial experiment evidence
        if progress_tracker is not None:
            progress_tracker.poll()
        child_returncode = 2
        _append_child_runner_error(child_output_path, exc, run_dir=run_dir)
        trace_writer.write(
            "child_agent_failed",
            phase="runner",
            summary="Mesh-segmentation child-agent runner failed.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={
                "error_type": type(exc).__name__,
                "error": str(exc),
                "last_observed_stage": (
                    progress_tracker.last_stage
                    if progress_tracker is not None
                    else None
                ),
                "last_observed_progress": (
                    progress_tracker.last_summary
                    if progress_tracker is not None
                    else None
                ),
            },
        )
    finally:
        if memory_broker is not None:
            memory_broker.close()
            trace_writer.write(
                "agent_memory_broker_stopped",
                phase="cleanup",
                summary="Stopped the wrapper-owned observation-memory broker.",
            )
        if parent_usd_cli_capability is not None:
            stop_parent_usd_cli_capability(parent_usd_cli_capability)

    _reject_unsafe_run_links(run_dir)
    required_artifacts = list(request["required_final_artifacts"])
    terminal = _validate_terminal_artifacts(
        run_dir,
        required_artifacts=required_artifacts,
        authoritative_request=request,
    )
    _write_json(terminal_validation_path, terminal)
    completed = child_returncode == 0 and bool(terminal["valid"])
    returncode = child_returncode
    if child_returncode == 0 and not completed:
        returncode = 1
    trace_writer.write(
        "terminal_validation",
        phase="validation",
        summary=(
            "Mesh-segmentation artifact contract passed."
            if terminal["valid"]
            else "Mesh-segmentation artifact contract is incomplete."
        ),
        artifacts=[str(terminal_validation_path)],
        data={
            "valid": terminal["valid"],
            "missing_artifacts": terminal["missing_artifacts"],
            "wall_time_seconds": time.monotonic() - run_started,
            "last_observed_stage": (
                progress_tracker.last_stage if progress_tracker is not None else None
            ),
            "last_observed_progress": (
                progress_tracker.last_summary if progress_tracker is not None else None
            ),
        },
    )
    return MeshSegmentationResult(
        run_dir=run_dir,
        request_path=request_path,
        prompt_path=prompt_path,
        child_output_path=child_output_path,
        child_final_path=child_final_path,
        terminal_validation_path=terminal_validation_path,
        returncode=returncode,
        completed=completed,
    )


def _validate_config(config: MeshSegmentationConfig) -> None:
    if config.workflow_skill not in SUPPORTED_WORKFLOW_SKILLS:
        raise ValueError(f"Unsupported --workflow-skill: {config.workflow_skill}")
    if config.runner not in SUPPORTED_RUNNERS:
        raise ValueError(f"Unsupported --runner: {config.runner}")
    if config.codex_sandbox_mode not in SUPPORTED_CODEX_SANDBOX_MODES:
        raise ValueError(
            f"Unsupported --codex-sandbox-mode: {config.codex_sandbox_mode}"
        )
    if config.codex_execution_mode not in SUPPORTED_CODEX_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported --codex-execution-mode: {config.codex_execution_mode}"
        )
    if config.case_timeout_seconds < 0:
        # `max(0.0, ...)` downstream turned a negative into the documented
        # "disable the cap" value, so a mistyped `-1` silently expanded a
        # bounded case to iteration_budget x child_timeout. Only an explicit
        # zero disables the cap.
        raise ValueError(
            "--case-timeout must be zero (disabled) or positive, got "
            f"{config.case_timeout_seconds}"
        )
    if config.workflow_skill == "content-workflow-mesh-segmentation":
        if config.runner != RUNNER_CODEX:
            raise ValueError("Mesh segmentation requires --runner=codex.")
        if config.codex_execution_mode != CODEX_EXECUTION_HOST:
            raise ValueError(
                "Mesh segmentation requires --codex-execution-mode=host; "
                "Docker/container execution is not allowed."
            )
        required_provider_values = {
            "--codex-responses-url": config.codex_responses_url,
            "--codex-api-key-env": config.codex_api_key_env,
        }
        if config.allow_codex_configured_auth:
            configured_auth_overrides = {
                "--codex-base-url": config.codex_base_url,
                **required_provider_values,
                "--codex-config-json/--codex-config-file": config.codex_config,
            }
            conflicting_overrides = [
                name for name, value in configured_auth_overrides.items() if value
            ]
            if conflicting_overrides:
                raise ValueError(
                    "--allow-codex-configured-auth cannot be combined with: "
                    + ", ".join(conflicting_overrides)
                )
        else:
            missing_provider_values = [
                name for name, value in required_provider_values.items() if not value
            ]
            if missing_provider_values:
                raise ValueError(
                    "Mesh segmentation requires explicit provider configuration or "
                    "--allow-codex-configured-auth: "
                    + ", ".join(missing_provider_values)
                )
            parsed_provider = urlparse(str(config.codex_responses_url))
            if (
                parsed_provider.scheme not in {"http", "https"}
                or not parsed_provider.netloc
                or not parsed_provider.path.rstrip("/").endswith("/responses")
                or parsed_provider.params
                or parsed_provider.query
                or parsed_provider.fragment
            ):
                raise ValueError(
                    "--codex-responses-url must be an absolute HTTP(S) URL ending "
                    "in /responses without params, query, or fragment."
                )
        backend_only_image_generation_options = {
            "--image-gen-base-url": config.image_gen_base_url,
            "--image-gen-api-key-env": config.image_gen_api_key_env,
        }
        if (
            any(backend_only_image_generation_options.values())
            and not config.image_gen_backend
        ):
            raise ValueError(
                "Image-generation endpoint or credential options require "
                "--image-gen-backend."
            )
        if config.image_gen_backend:
            from world_understanding.registry import (
                get_image_generation_model_registry,
            )

            registered_image_backends = set(
                get_image_generation_model_registry().list_models()
            )
            supported_image_backends = registered_image_backends | {"openai_compatible"}
            if config.image_gen_backend not in supported_image_backends:
                raise ValueError(
                    "Unsupported --image-gen-backend: "
                    f"{config.image_gen_backend!r}. Registered backends: "
                    f"{sorted(supported_image_backends)}"
                )
        if config.image_gen_backend == "openai_compatible":
            required_compatibility_values = {
                "--image-gen-model": config.image_gen_model,
                "--image-gen-base-url": config.image_gen_base_url,
                "--image-gen-api-key-env": config.image_gen_api_key_env,
            }
            missing_image_values = [
                name
                for name, value in required_compatibility_values.items()
                if not value
            ]
            if missing_image_values:
                raise ValueError(
                    "--image-gen-backend=openai_compatible requires: "
                    + ", ".join(missing_image_values)
                )
        if config.image_gen_base_url:
            parsed_image_gen_base = urlparse(config.image_gen_base_url)
            if (
                parsed_image_gen_base.scheme not in {"http", "https"}
                or not parsed_image_gen_base.netloc
                or parsed_image_gen_base.params
                or parsed_image_gen_base.query
                or parsed_image_gen_base.fragment
                or parsed_image_gen_base.path.rstrip("/").endswith(
                    ("/images/generations", "/chat/completions")
                )
            ):
                raise ValueError(
                    "--image-gen-base-url must be an absolute OpenAI-compatible "
                    "HTTP(S) base URL, not a leaf endpoint, and must not contain "
                    "params, query, or fragment."
                )
        key_env_fields = []
        if not config.allow_codex_configured_auth:
            key_env_fields.append(("--codex-api-key-env", config.codex_api_key_env))
        if config.image_gen_api_key_env:
            key_env_fields.append(
                ("--image-gen-api-key-env", config.image_gen_api_key_env)
            )
        for label, value in key_env_fields:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(value)):
                raise ValueError(f"{label} must be an environment-variable name.")
    if config.continue_from_run is not None:
        if config.workflow_skill != "content-workflow-mesh-segmentation":
            raise ValueError(
                "--continue-from-run requires "
                "--workflow-skill=content-workflow-mesh-segmentation."
            )
        source_run = config.continue_from_run.resolve()
        if not source_run.is_dir():
            raise ValueError(f"Continuation run is not a directory: {source_run}")
        _continuation_source_mode(source_run)
    if config.codex_execution_mode == CODEX_EXECUTION_CONTAINER:
        if config.runner != RUNNER_CODEX:
            raise ValueError(
                "--codex-execution-mode=container requires --runner=codex."
            )
        if not config.codex_container_image.strip():
            raise ValueError("--codex-container-image must not be empty.")
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is required for --codex-execution-mode=container."
            )
        auth_path = (
            _effective_codex_home(child_cwd=config.agent_cwd or config.repo_root)
            / "auth.json"
        )
        if not auth_path.is_file():
            raise FileNotFoundError(
                "Containerized Codex requires a file-backed Codex login at "
                f"{auth_path}. Run `content-workflow-cli auth login` first."
            )
    if config.codex_sandbox_mode == CODEX_SANDBOX_DANGER_FULL_ACCESS:
        if config.codex_execution_mode != CODEX_EXECUTION_HOST:
            raise ValueError(
                "--codex-sandbox-mode=danger-full-access requires host execution."
            )
        if not config.allow_unsafe_host_child:
            raise ValueError(
                "--codex-sandbox-mode=danger-full-access requires "
                "--allow-unsafe-host-child."
            )
    if config.claude_execution_mode not in SUPPORTED_CLAUDE_EXECUTION_MODES:
        raise ValueError(
            f"Unsupported --claude-execution-mode: {config.claude_execution_mode}"
        )
    if config.claude_max_turns is not None and config.claude_max_turns <= 0:
        raise ValueError("--claude-max-turns must be greater than 0.")
    if config.child_timeout_seconds < 0:
        raise ValueError("--child-timeout must be greater than or equal to 0.")
    if config.scene_tool_timeout_seconds <= 0:
        raise ValueError("--scene-tool-timeout must be greater than 0.")
    if config.iteration_budget <= 0:
        raise ValueError("--iteration-budget must be greater than 0.")
    if not config.asset_path.is_file():
        raise FileNotFoundError(f"Input asset does not exist: {config.asset_path}")
    if config.asset_path.suffix.lower() not in SUPPORTED_ASSET_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_ASSET_SUFFIXES))
        raise ValueError(f"Input asset must use one of: {supported}")
    normalized_parts = [part.strip() for part in config.target_semantic_parts]
    if any(not part for part in normalized_parts):
        raise ValueError("--target-semantic-part must not be empty.")
    if any(Path(part).name != part for part in normalized_parts):
        raise ValueError(
            "--target-semantic-part values must be plain semantic names, not paths."
        )
    folded_parts = [part.casefold() for part in normalized_parts]
    if len(set(folded_parts)) != len(folded_parts):
        raise ValueError("--target-semantic-part values must be unique.")
    # Casefold uniqueness is not enough: gating keys parts by the directory
    # fold, where `a-b` and `a_b`, or `01_panel` and `panel`, become one key.
    # Two legal targets would then share a key and `setdefault` would silently
    # drop the second from initializer gating while their evidence aliased.
    # Fail up front instead of losing a requested target mid-run.
    directory_keys: dict[str, list[str]] = {}
    for part in normalized_parts:
        directory_keys.setdefault(_normalize_part_key(part), []).append(part)
    collisions = sorted(
        ", ".join(parts) for parts in directory_keys.values() if len(parts) > 1
    )
    if collisions:
        raise ValueError(
            "--target-semantic-part values must stay distinct once folded to "
            f"directory names: {'; '.join(collisions)}"
        )
    reserved_parts = sorted(
        part
        for part, folded in zip(normalized_parts, folded_parts, strict=True)
        if folded in RESERVED_TARGET_SEMANTIC_PART_NAMES
    )
    if reserved_parts:
        raise ValueError(
            "--target-semantic-part values conflict with reserved run artifact "
            f"names: {', '.join(reserved_parts)}"
        )
    if config.asset_root is not None:
        if not config.asset_root.is_dir():
            raise ValueError(f"--asset-root is not a directory: {config.asset_root}")
        try:
            config.asset_path.relative_to(config.asset_root)
        except ValueError as exc:
            raise ValueError(
                f"Input asset is outside --asset-root: {config.asset_path}"
            ) from exc
    if not config.target_semantic_parts and not config.reference_images:
        raise ValueError(
            "At least one --reference-image or --reference-dir is required."
        )
    for path in config.reference_images:
        if not path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {path}")
    if config.expected_asset_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", config.expected_asset_sha256
    ):
        raise ValueError("--expected-asset-sha256 must be a lowercase SHA-256 digest")
    if config.expected_reference_sha256 and len(
        config.expected_reference_sha256
    ) != len(config.reference_images):
        raise ValueError(
            "--expected-reference-sha256 count must match the expanded reference "
            "image count"
        )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in config.expected_reference_sha256
    ):
        raise ValueError(
            "--expected-reference-sha256 values must be lowercase SHA-256 digests"
        )
    if config.target_prim_path and not config.target_prim_path.startswith("/"):
        raise ValueError("--target-prim must be an absolute USD prim path.")
    skill = (
        config.repo_root
        / "agentic"
        / ".agents"
        / "skills"
        / config.workflow_skill
        / "SKILL.md"
    )
    if not skill.is_file():
        raise FileNotFoundError(f"Mesh-segmentation workflow skill is missing: {skill}")


def _resolve_run_dir(config: MeshSegmentationConfig) -> tuple[str, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    prefix = "mesh-segmentation-"
    slug_budget = 128 - len(prefix) - len(stamp) - 1
    default_run_id = f"{prefix}{_slug(config.asset_path.stem)[:slug_budget]}-{stamp}"
    run_id = (config.run_id or default_run_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "--run-id must start with an alphanumeric character and contain only "
            "letters, numbers, dot, underscore, or hyphen."
        )
    if len(run_id) > 128:
        raise ValueError("--run-id must be at most 128 characters.")
    candidate = (
        config.output_dir
        if config.output_dir is not None
        else config.repo_root / "runs" / run_id
    )
    _reject_unsafe_run_links(candidate, allow_missing=True)
    run_dir = _lexical_absolute_path(candidate)
    if run_dir.exists():
        raise FileExistsError(
            f"A fresh mesh-segmentation run requires a new output directory: {run_dir}"
        )
    return run_id, run_dir


def _initialize_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    _reject_unsafe_run_links(run_dir)
    for relative in ("inputs/source", "inputs/references", "raw", "trace"):
        path = run_dir / relative
        path.mkdir(parents=True)
        _chmod_private(path)
    _chmod_private(run_dir)


def _prepare_run_dir(config: MeshSegmentationConfig) -> tuple[str, Path]:
    """Resolve and initialize a fresh run directory for tests and callers."""

    run_id, run_dir = _resolve_run_dir(config)
    _initialize_run_dir(run_dir)
    return run_id, run_dir


def _continuation_source_mode(source_run: Path) -> str:
    """Require a completed run whose terminal contracts passed."""

    terminal, _ = _load_json_object(
        source_run / "terminal_validation.json",
        label="continuation terminal_validation.json",
    )
    final_validation, _ = _load_json_object(
        source_run / "validation" / "final_validation.json",
        label="continuation validation/final_validation.json",
    )
    if (
        terminal is not None
        and terminal.get("valid") is True
        and final_validation is not None
        and final_validation.get("status") == "passed"
    ):
        return "validated_final"
    raise ValueError(
        "Continuation requires a completed run with passing "
        "terminal_validation.json and validation/final_validation.json."
    )


def _stage_inputs(
    config: MeshSegmentationConfig,
    run_dir: Path,
) -> tuple[Path, list[Path], dict[str, object]]:
    source_root = run_dir / "inputs" / "source"
    if config.asset_root is not None:
        bundle_root = source_root / "asset_bundle"
        # Preserve links as links so staging never dereferences an asset-bundle
        # symlink into an arbitrary host path. The run-tree validator below then
        # rejects every copied link before any child process can consume it.
        shutil.copytree(config.asset_root, bundle_root, symlinks=True)
        staged_asset = bundle_root / config.asset_path.relative_to(config.asset_root)
        staging_mode = "asset_bundle_copy"
    else:
        staged_asset = source_root / _safe_filename(config.asset_path.name, "asset")
        shutil.copy2(config.asset_path, staged_asset)
        staging_mode = "single_file_copy"
    _reject_unsafe_run_links(run_dir)
    staged_asset_sha256 = _sha256_file(staged_asset)
    if (
        config.expected_asset_sha256 is not None
        and staged_asset_sha256 != config.expected_asset_sha256
    ):
        raise ValueError(
            "Staged mesh-segmentation asset does not match --expected-asset-sha256"
        )

    staged_references: list[Path] = []
    reference_records: list[dict[str, object]] = []
    reference_root = run_dir / "inputs" / "references"
    for index, source in enumerate(config.reference_images, start=1):
        filename = _safe_filename(source.name, f"reference_{index:02d}")
        destination = reference_root / f"{index:02d}_{filename}"
        shutil.copy2(source, destination)
        staged_reference_sha256 = _sha256_file(destination)
        if config.expected_reference_sha256:
            expected_sha256 = config.expected_reference_sha256[index - 1]
            if staged_reference_sha256 != expected_sha256:
                raise ValueError(
                    "Staged mesh-segmentation reference image "
                    f"{index} does not match --expected-reference-sha256"
                )
        staged_references.append(destination)
        reference_records.append(
            {
                "source_name": source.name,
                "staged_path": str(destination),
                "sha256": staged_reference_sha256,
                "expected_sha256": (
                    config.expected_reference_sha256[index - 1]
                    if config.expected_reference_sha256
                    else None
                ),
                "size_bytes": destination.stat().st_size,
            }
        )
    _reject_unsafe_run_links(run_dir)
    return (
        staged_asset,
        staged_references,
        {
            "mode": staging_mode,
            "asset": {
                "source_name": config.asset_path.name,
                "staged_path": str(staged_asset),
                "sha256": staged_asset_sha256,
                "expected_sha256": config.expected_asset_sha256,
                "size_bytes": staged_asset.stat().st_size,
            },
            "reference_images": reference_records,
        },
    )


def _stage_continuation_seed(
    *,
    source_run: Path,
    run_dir: Path,
    staged_asset: Path,
) -> dict[str, object]:
    source = source_run.resolve()
    source_mode = _continuation_source_mode(source)
    staged_asset_sha256 = _sha256_file(staged_asset)
    export_manifest, export_error = _load_json_object(
        source / "final" / "export_manifest.json",
        label="continuation final/export_manifest.json",
    )
    if export_error:
        raise ValueError(export_error)
    assert export_manifest is not None
    source_asset_sha256 = export_manifest.get("source_sha256")
    if export_manifest.get("exact_source_face_coverage") is not True:
        raise ValueError("Continuation seed lacks exact source-face coverage.")
    seed_labels_path = source / "state" / "final_labels.u32le"
    if source_asset_sha256 != staged_asset_sha256:
        raise ValueError(
            "Continuation seed was produced from a different source asset."
        )

    fragment_labels_path = source / "fragments" / "fragment_ids.u32le"
    seed_labels, seed_labels_error = _load_u32_labels(seed_labels_path)
    fragment_labels, fragment_labels_error = _load_u32_labels(fragment_labels_path)
    if seed_labels_error:
        raise ValueError(seed_labels_error)
    if fragment_labels_error:
        raise ValueError(fragment_labels_error)
    assert seed_labels is not None
    assert fragment_labels is not None
    if len(seed_labels) != len(fragment_labels):
        raise ValueError(
            "Continuation semantic and fragment label counts do not match."
        )
    locked_segment_ids = sorted({value for value in seed_labels if value != 0})
    if not locked_segment_ids:
        raise ValueError("Continuation seed contains no locked semantic labels.")
    fragment_semantics: dict[int, int] = {}
    for fragment_id, semantic_id in zip(
        fragment_labels,
        seed_labels,
        strict=True,
    ):
        prior = fragment_semantics.setdefault(fragment_id, semantic_id)
        if prior != semantic_id:
            raise ValueError(
                f"Continuation seed splits immutable fragment {fragment_id}."
            )

    segment_manifest, segment_manifest_error = _load_json_object(
        source / "segments.json",
        label="continuation segments.json",
    )
    if segment_manifest_error:
        raise ValueError(segment_manifest_error)
    assert segment_manifest is not None
    raw_parts = segment_manifest.get("segments")
    if not isinstance(raw_parts, list):
        raise ValueError("Continuation segments.json must contain a segments array.")
    raw_segments = [
        {
            "segment_id": part.get("segment_id"),
            "name": part.get("name"),
        }
        for part in raw_parts
        if isinstance(part, dict)
        and isinstance(part.get("segment_id"), int)
        and isinstance(part.get("name"), str)
    ]
    segment_names = {
        int(segment["segment_id"]): str(segment["name"])
        for segment in raw_segments
        if isinstance(segment, dict)
        and isinstance(segment.get("segment_id"), int)
        and isinstance(segment.get("name"), str)
    }
    missing_names = [
        segment_id
        for segment_id in locked_segment_ids
        if segment_id not in segment_names
    ]
    if missing_names:
        raise ValueError(
            f"Continuation seed lacks names for locked segment IDs: {missing_names}"
        )

    fixed_artifacts = (
        "prepare/neutral.usdc",
        "prepare/topology.json",
        "prepare/all_faces_candidate.u32le",
        "prepare/degenerate_face_ids.u32le",
        "fragments/fragment_ids.u32le",
        "fragments/fragment_ids.npy",
        "fragments/fragment_adjacency.npy",
        "fragments/fragment_statistics.npz",
        "fragments/fragment_colors.npy",
        "fragments/fragments.usdc",
        "fragments/fragment_manifest.json",
    )
    for relative in fixed_artifacts:
        source_path = source / relative
        if not source_path.is_file():
            raise ValueError(f"Continuation artifact is missing: {relative}")
        destination = run_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        _chmod_private(destination)

    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    parent_labels_path = state_dir / "labels-parent.u32le"
    shutil.copy2(seed_labels_path, parent_labels_path)
    _chmod_private(parent_labels_path)
    seed_dir = run_dir / "continuation_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    immutable_labels_path = seed_dir / "immutable_face_labels.u32le"
    shutil.copy2(seed_labels_path, immutable_labels_path)
    _chmod_private(immutable_labels_path)
    source_segments_path = source / "segments.json"
    if source_segments_path.is_file():
        shutil.copy2(source_segments_path, seed_dir / "seed_segments.json")
    else:
        _write_json(
            seed_dir / "seed_segments.json",
            {
                "schema_version": "mesh-segmentation.segments.v1",
                "segments": raw_segments,
            },
        )
    shutil.copy2(source / "part_plan.json", seed_dir / "seed_part_plan.json")
    shutil.copy2(
        source / "validation" / "final_validation.json",
        seed_dir / "source_final_validation.json",
    )
    shutil.copy2(
        source / "terminal_validation.json",
        seed_dir / "source_terminal_validation.json",
    )
    in_progress_part: dict[str, object] | None = None

    locked_segments: list[dict[str, object]] = []
    for segment_id in locked_segment_ids:
        name = segment_names[segment_id]
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(
                f"Continuation segment {segment_id} has an unsafe name: {name!r}."
            )
        # Resolve the same way terminal validation does. Looking only for the
        # exact semantic name meant a run using the documented `parts/`-nested,
        # `NN_`-prefixed, or slugged layout passed every terminal contract and
        # then failed `--continue-from-run` as missing its completion.
        resolved_part_dir = _resolve_part_dir(source, name)
        if resolved_part_dir is not None:
            completion_path = resolved_part_dir / "part_completion.json"
        else:
            completion_relative = f"{name}/part_completion.json"
            completion_path = _safe_run_artifact_path(source, completion_relative)
        if completion_path is None:
            raise ValueError(
                f"Continuation completion record is unsafe for locked part {name}."
            )
        if not completion_path.is_file():
            raise ValueError(
                f"Continuation part completion is missing for locked part {name}."
            )
        completion, completion_error = _load_json_object(
            completion_path,
            label=f"continuation completion for {name}",
        )
        if completion_error:
            raise ValueError(completion_error)
        assert completion is not None
        expected_completion_fields = {
            "schema_version": "mesh-segmentation-part-completion.v1",
            "status": "locked",
            "semantic_part": name,
            "segment_id": segment_id,
        }
        for field_name, expected_value in expected_completion_fields.items():
            if completion.get(field_name) != expected_value:
                raise ValueError(
                    f"Continuation completion for {name} requires "
                    f"{field_name}={expected_value!r}."
                )
        expected_face_count = sum(value == segment_id for value in seed_labels)
        recorded_face_count = completion.get(
            "locked_face_count",
            completion.get("selected_face_count"),
        )
        if (
            isinstance(recorded_face_count, int)
            and recorded_face_count != expected_face_count
        ):
            raise ValueError(
                f"Continuation face count for locked part {name} does not match "
                "state/labels-parent.u32le."
            )
        copied_completion = seed_dir / "part_completions" / f"{name}.json"
        copied_completion.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(completion_path, copied_completion)
        locked_segments.append(
            {
                "segment_id": segment_id,
                "name": name,
                "face_count": expected_face_count,
                "part_completion": str(copied_completion),
                "part_completion_sha256": _sha256_file(copied_completion),
            }
        )

    manifest = {
        "schema_version": "content-agents.mesh-segmentation-continuation-seed.v1",
        "source_mode": source_mode,
        "source_run_id": source.name,
        "source_asset_sha256": staged_asset_sha256,
        "source_face_count": len(seed_labels),
        "fragment_count": len(fragment_semantics),
        "fragment_labels": str(run_dir / "fragments" / "fragment_ids.u32le"),
        "fragment_labels_sha256": _sha256_file(
            run_dir / "fragments" / "fragment_ids.u32le"
        ),
        "immutable_face_labels": str(immutable_labels_path),
        "immutable_face_labels_sha256": _sha256_file(immutable_labels_path),
        "parent_labels": str(parent_labels_path),
        "parent_labels_sha256": _sha256_file(parent_labels_path),
        "locked_segments": locked_segments,
        "locked_segment_ids": locked_segment_ids,
        "in_progress_part": in_progress_part,
        "background_segment_id": 0,
        "prior_run_access_allowed": False,
        "seed_artifacts_copied_into_permitted_evidence_root": True,
    }
    manifest_path = seed_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _reject_unsafe_run_links(run_dir)
    return {
        "schema_version": manifest["schema_version"],
        "source_run_id": source.name,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "locked_segments": locked_segments,
        "in_progress_part": in_progress_part,
    }


def _build_request(
    config: MeshSegmentationConfig,
    *,
    run_id: str,
    run_dir: Path,
    staging: dict[str, object],
    memory_root: Path | None = None,
    memory_broker_url: str | None = None,
) -> dict[str, object]:
    targeted = bool(config.target_semantic_parts)
    explicit_image_generation = bool(config.image_gen_backend)
    image_generation_mode = (
        "world_understanding_backend"
        if explicit_image_generation
        else "coding_agent_companion"
    )
    workflow_mode = "targeted" if targeted else "recognition"
    required_artifacts = list(
        required_mesh_segmentation_artifacts(
            workflow_mode,
            resumable=config.continue_from_run is not None,
        )
    )
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "created_at": utc_now(),
        "workflow": "mesh-segmentation.run",
        "workflow_mode": workflow_mode,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dry_run": config.dry_run,
        "required_skills": list(MESH_SEGMENTATION_REQUIRED_SKILLS),
        "inputs": {
            "asset": str(config.asset_path),
            "target_prim": config.target_prim_path,
            "target_semantic_parts": list(config.target_semantic_parts),
            "reference_images": [str(path) for path in config.reference_images],
            "continuation_seed": staging.get("continuation"),
        },
        "input_staging": staging,
        "isolation": {
            "fresh_child_thread": True,
            "conversation_context_inherited": False,
            "prior_run_access_allowed": False,
            "working_directory": str(run_dir),
            "permitted_evidence_root": str(run_dir),
            "permitted_memory_root": None,
            "memory_access": ("launcher_broker" if config.memory_enabled else None),
        },
        "constraints": {
            "source_asset_edits_allowed": False,
            "segmentation_strategy": (
                "immutable_fragment_direct_visual_include_exclude"
            ),
            "oversegmentation_strategy": ("boundary_safe_fast_mesh_superfacets_v1"),
            "semantic_decision_unit": "immutable_fragment",
            "triangle_level_semantic_edits_allowed": False,
            "require_fragment_atomicity": True,
            "initialization_strategy": (
                "per_part_registered_mask_probe_router_v1"
                if explicit_image_generation
                else "per_part_companion_mask_probe_or_warned_direct_v1"
            ),
            "initialization_marker_policy": (
                "three_view_mask_probe_then_recorded_route"
                if explicit_image_generation
                else "try_companion_three_view_probe_else_warned_direct"
            ),
            "initialization_multi_view_reduction": (
                "mask_route_exact_union_direct_route_none"
            ),
            "initialization_view_policy": (
                "three_view_probe_then_eight_mask_or_adaptive_direct"
                if explicit_image_generation
                else "companion_three_view_probe_if_available_else_adaptive_direct"
            ),
            "initialization_visibility_policy": "closest_ray_intersection_only",
            "initialization_mesh_expansion": ("none_direct_fragment_decisions"),
            "image_generation_allowed": True,
            "image_generation_mode": image_generation_mode,
            "image_generation_unavailable_policy": (
                "warn_and_use_direct_agentic_selection"
            ),
            "semantic_mask_projection_allowed": True,
            "pixel_or_view_voting_allowed": False,
            "semantic_masks_are_authoritative": False,
            "require_initializer_decision": True,
            "initializer_decisions": (["image_mask_seed", "direct_agentic_selection"]),
            "initializer_revision_is_provisional": True,
            "require_falsification_plan": True,
            "require_signed_confuser_evidence": True,
            "require_boundary_challenge_coverage": True,
            "require_selected_only_held_out_review": True,
            "require_per_instance_selected_only_completeness": True,
            "require_unexplained_selected_boundary_zero": True,
            "require_adjacent_unselected_frontier_challenge": True,
            "require_falsification_validation": True,
            "auxiliary_render_channels": (
                [
                    "normal",
                    "linear_depth",
                    "closest_visible_face_id",
                    "closest_visible_fragment_id",
                    "flat_binary_label",
                ]
            ),
            "require_signed_fragment_evidence": True,
            "require_hypothesis_card": False,
            "require_false_positive_review": True,
            "require_false_negative_review": True,
            "require_same_camera_id_mapping": True,
            "anti_oscillation_revision_limit": None,
            "anti_oscillation_policy": (
                "progress_sensitive_retry_else_defer_continue_and_revisit"
            ),
            "one_active_logical_part_at_a_time": True,
            "recognition_queue_order": (
                None if targeted else "largest_reliable_distinctive_to_small_body_last"
            ),
            "unstarted_parts_must_have_zero_faces": True,
            "locked_revisions_are_immutable": True,
            "locked_fragment_reassignment_requires_transactional_supersession": False,
            "continuation_seed_required": config.continue_from_run is not None,
            "require_part_plan": True,
            "iteration_budget": config.iteration_budget,
            "base_evidence_view_policy": "exact_eight_cube_corners",
            "require_multi_view_validation": True,
            "require_exact_face_provenance": True,
            "category_specific_constants_allowed": False,
            "part_recognition_required": not targeted,
            "output_partition": (
                "target_parts_plus_other"
                if targeted
                else "all_recognized_material_families"
            ),
            "require_pixel_to_face_seed_events": True,
            "require_held_out_visual_validation": True,
            "require_agent_memory": config.memory_enabled,
            "residual_body_last": True,
        },
        "runtime": {
            "runner": config.runner,
            "model": config.model,
            "model_reasoning_effort": config.model_reasoning_effort,
            "codex_responses_url": config.codex_responses_url,
            "codex_sdk_base_url": config.codex_base_url,
            "codex_api_key_env": config.codex_api_key_env,
            "codex_auth_mode": (
                "configured_sdk"
                if config.allow_codex_configured_auth
                else "explicit_provider"
            ),
            "python_executable": (
                "/usr/local/bin/python3"
                if config.codex_execution_mode == CODEX_EXECUTION_CONTAINER
                else sys.executable
            ),
            "codex_execution_mode": config.codex_execution_mode,
            "allow_unsafe_host_child": config.allow_unsafe_host_child,
            "codex_container_image": (
                config.codex_container_image
                if config.codex_execution_mode == CODEX_EXECUTION_CONTAINER
                else None
            ),
            "scene_tool_timeout_seconds": config.scene_tool_timeout_seconds,
            "usd_cli_session_id": config.usd_cli_session_id,
            "child_timeout_seconds": config.child_timeout_seconds,
            "agent_memory": {
                "enabled": config.memory_enabled,
                "required_capture": config.memory_enabled,
                "run_id": run_id if config.memory_enabled else None,
                "root": str(memory_root) if memory_root is not None else None,
                "broker_url": memory_broker_url,
                "access": "launcher_broker" if config.memory_enabled else None,
                "scope": "current_run" if config.memory_enabled else None,
                "context_limit": 12 if config.memory_enabled else None,
            },
            "image_generation": {
                "mode": image_generation_mode,
                "backend": (
                    config.image_gen_backend if explicit_image_generation else None
                ),
                "model": config.image_gen_model,
                "base_url": config.image_gen_base_url,
                "transport": (
                    "world_understanding_image_generation_model"
                    if explicit_image_generation
                    else "coding_agent_companion_tool"
                ),
                "api_key_env": (
                    config.image_gen_api_key_env if explicit_image_generation else None
                ),
                "fallback": "warn_and_use_direct_agentic_selection",
                "warning_artifact": ("PART/image_generation_warning.json"),
            },
        },
        "required_final_artifacts": required_artifacts,
        "additional_instructions": (
            config.additional_instructions.strip()
            if config.additional_instructions and config.additional_instructions.strip()
            else None
        ),
    }


def _bind_mesh_helper_paths(
    prompt: str,
    helper_scripts: dict[str, Path],
) -> str:
    """Bind prompt helper references to checked-in, sandbox-read-only files."""

    referenced = set(re.findall(r"scripts/([A-Za-z0-9_.-]+)", prompt))
    missing = sorted(referenced - helper_scripts.keys())
    if missing:
        raise ValueError(
            "Mesh-segmentation prompt references unavailable helper scripts: "
            + ", ".join(missing)
        )
    for name in sorted(referenced, key=len, reverse=True):
        prompt = prompt.replace(f"scripts/{name}", str(helper_scripts[name]))
    for name, helper_path in helper_scripts.items():
        prompt = prompt.replace(f"`{name}`", f"`{helper_path}`")
    return prompt


def _build_prompt(
    config: MeshSegmentationConfig,
    *,
    request_path: Path,
    run_dir: Path,
    run_id: str,
    memory_root: Path | None,
    memory_broker_url: str | None,
    helper_scripts: dict[str, Path],
) -> str:
    if config.target_semantic_parts:
        targets = ", ".join(f"`{part}`" for part in config.target_semantic_parts)
        target_mode = (
            f"The user-provided semantic target vocabulary is exactly: {targets}. "
            "Skip part recognition. Write `part_plan.json` using exactly these "
            "semantic names—do not add, rename, split, or merge non-residual "
            "categories. Create each part directory directly under the run "
            "directory with that exact case-sensitive string (for example, "
            "`Base/` and `Fan Blades/`); never lowercase, slug, number, or nest "
            "target directories. Record the identical string in every "
            "`semantic_part` field. "
            "Use visual inspection only to record instances, confusers, and "
            "descriptions, choose a safe processing order with broad provided "
            "body categories last, then process one vocabulary part at a time. "
            "Export those targets plus `other` only as the residual for any "
            "faces left unassigned."
        )
    else:
        target_mode = (
            "Inspect the staged references and neutral renders, write the complete "
            "semantic queue to `part_plan.json`, and process one part at a time."
        )
    if config.target_semantic_parts:
        part_directory_contract = (
            "`PART` is one directory per requested semantic target, directly under "
            "the run directory, using the exact case-sensitive target string. "
            "Do not lowercase, slug, number, or nest targeted part directories."
        )
    else:
        part_directory_contract = (
            "`PART` is one directory per recognized semantic part, directly under "
            "the run directory, named so it folds to that part's semantic name: "
            "lowercase, with every run of non-alphanumeric characters collapsed "
            "to `_`. `Car Chassis` becomes `car_chassis`. An optional `NN_` order "
            "prefix is allowed (`08_car_chassis`), as is collecting the directories "
            "under `parts/`."
        )
    continuation_mode = ""
    if config.continue_from_run is not None:
        continuation_mode = f"""
Validated continuation seed:
- Read `{run_dir / "continuation_seed" / "manifest.json"}`.
- Reuse the already staged `prepare/` and `fragments/` artifacts byte-for-byte;
  do not prepare the mesh again and do not regenerate or renumber fragments.
- Start from `state/labels-parent.u32le`. Every nonzero seed label is a
  user-accepted completed part and is immutable.
- Include each seeded part in the new part plan as already completed, but do
  not run its initializer or refinement again. Recognize and segment only the
  remaining visible parts, then replace the old residual with the completed
  multi-part export.
- Never read the source run or any sibling run. The copied continuation seed is
  the complete permitted prior evidence.
"""
    locked_state_resume_contract = (
        "Every accepted lock is immutable: never extend, reassign, or otherwise "
        "change its faces."
    )
    locked_revision_contract = """
- A validated lock, its faces, prior revisions, and evidence are immutable in
  every run mode. Never extend, reassign, or otherwise change a face belonging
  to an accepted lock. Continue only from unlocked, provisional, or residual
  faces.
- Do not lock a part until a complete post-edit multi-view review records both
  `false_positive_review: passed` and `false_negative_review: passed`.
"""
    if config.memory_enabled:
        assert memory_root is not None
        broker_url = memory_broker_url or "<launcher-memory-broker-url>"
        memory_cli_path = Path(sys.executable).with_name("content-agent-memory")
        memory_cli_command = (
            str(memory_cli_path)
            if memory_cli_path.is_file()
            else "content-agent-memory"
        )
        memory_contract = f"""
Observation-memory contract:
- Memory is required for this run. Use only `{memory_cli_command}`; never scan
  its object store or SQLite index directly.
- Use run ID `{run_id}` and broker URL `{broker_url}` for every memory command.
  The durable root `{memory_root}` is wrapper-owned and must never be read or
  written directly.
- Memory is a checkpoint, not a second decision-maker. Keep the semantic
  evidence and quality bar identical to a memory-disabled run.
- Current-run memory is empty before the initial plan and before a part's first
  attempt, so do not query it then. Search only when resuming or deliberately
  revisiting an active part:
  `{memory_cli_command} --run-id {run_id} --broker-url {broker_url} search
  --workflow mesh-segmentation --target EXACT_PART --limit 4`. Search a named
  confuser separately only when relevant; never load unrelated part cards.
- Do not search memory before every revision or lock, and do not reload an
  unchanged observation. Memory capture is post-decision bookkeeping: first
  create the same candidate you would create with memory disabled.
  The launcher automatically records an evidence-only checkpoint when that
  exact revision has deterministic selected-only and frontier evidence; do not
  duplicate it or wait until the end. It deliberately makes no semantic claim. Record a
  materially changed diagnosis or correction, a deferral, or a validated lock
  yourself; do not record routine successful commands or repeat an unchanged
  backend warning.
- Final export is forbidden while any requested part lacks selected-only
  evidence, a frontier audit, or its launcher checkpoint. Finish the missing
  review first; a first-pass revision is not an export-ready result.
- A memory summary is not semantic evidence. Never select, reject, transfer, or
  preserve geometry merely to agree with a previous observation. On revisit,
  use memory to recover the prior candidate and unresolved question, then let
  the current signed and visual evidence decide whether to keep or change it.
  Use absolute artifact paths and chain changed decisions with
  `parent_observation_id`.
- Classify an initializer or unvalidated revision as `not_checked`, `ambiguous`,
  or `contradicted`. Use `matched` and pinning only after both false-positive
  and false-negative validation passes.
- Memory `interaction` accepts `operation`, optional request/response event IDs,
  `target_object_ids`, and `target_prim_paths` only. Put the exact semantic part
  in `target_object_ids` and tags; never add a `semantic_part` interaction field.
- Memory must not reduce cameras, renders, signed evidence, revisions, or
  inspection. Dense labels and validations remain authoritative. If required
  capture fails, retry it once, continue the segmentation, report the missing
  capture, and do not claim that the memory contract passed.
"""
    else:
        memory_contract = """
Observation-memory contract:
- Memory is disabled for this run. Do not claim durable observation recall.
"""
    if config.workflow_skill == "content-workflow-mesh-segmentation":
        if config.image_gen_backend:
            image_initializer_contract = """
- The frozen request explicitly configures a World Understanding image backend.
  Use the `image-generation` skill through the compatibility wrapper
  `scripts/generate_semantic_overlays.py` to generate the three representative cube-corner views
  as semantic-overlay probes. Preserve each shared
  image-generation manifest and affine-register them, then render matching
  closest-visible ID buffers and create the 2 px-eroded diagnostic fragment
  projection. Do not apply this three-view probe.
"""
        else:
            model_preference = (
                f"The requested model preference is `{config.image_gen_model}`; "
                "honor it only if the companion capability supports model selection."
                if config.image_gen_model
                else "No companion image-generation model is pinned."
            )
            image_initializer_contract = f"""
- No World Understanding image backend is configured. Use the `image-generation`
  skill, which defaults to the image-generation or image-editing capability
  accompanying this coding-agent session. {model_preference} Use it to edit the
  three representative cube-corner views into semantic overlays while preserving
  the neutral source cameras. Preserve its common image-generation manifests and
  raw outputs, affine-register the overlays, then render matching closest-visible
  ID buffers and create the 2 px-eroded diagnostic fragment projection. Do not
  apply this three-view probe.
- If no companion image-generation capability is available, or it cannot
  produce usable conditioned overlays, do not block and do not fabricate mask
  evidence. Write `PART/image_generation_warning.json` with schema
  `mesh-segmentation-image-generation-warning.v1`, severity `warning`, code
  `image_generation_unavailable`, a concrete reason, and fallback
  `direct_agentic_selection`. Report the warning in the final report. Record
  `probe_status: skipped_image_generation_unavailable` in
  `PART/initializer_decision.json`, validate it, and initialize that part
  directly from focused neutral/ID evidence.
"""
        return _bind_mesh_helper_paths(
            f"""Use the `content-workflow-mesh-segmentation`, `image-generation`,
and `usd-cli` skills to complete a fresh, blind mesh-segmentation run.

Frozen request: `{request_path}`
Run directory and complete permitted evidence corpus: `{run_dir}`

Read the frozen request and every staged reference image before acting.
The provider may have restarted this turn after a transient capacity error, so
inspect the run directory before creating or regenerating anything. Treat every
existing `part_completion.json` with `status: locked` as sealed: do not rewrite
that part's `initializer_decision.json`, initializer validation, `rev-000`, or
earlier evidence. Resume only unfinished work from the newest numbered state.
{locked_state_resume_contract}
{target_mode}
{continuation_mode}
{memory_contract}

Execution contract:
- Work natively in this fresh host session. Never start Docker.
{CONTROLLED_JSON_ARTIFACT_WRITE}
- A rejected write command is not evidence that the run directory is read-only.
  Use the controlled JSON procedure above, and let the checked-in helpers create
  their own output directories and binary artifacts.
- Write every file you create inside the run directory, including scratch
  inputs such as edit batches, click batches, and region marks. A manifest that
  references a path outside the run directory (for example under `/tmp`) is
  rejected as unsafe, because evidence outside the run cannot be audited.
- Do not inspect parent directories, sibling runs, git history, golden labels,
  prior experiments, or prior run summaries.
- Write this exact run layout. Every consumer of this run -- the launcher's
  gates and terminal validation -- locates evidence by these paths, so an
  invented alternative is not a stylistic choice: it makes correct work
  unreadable and fails the run.

  ```
  part_plan.json
  segments.json
  prepare/  fragments/  state/
  PART/                                   one directory per semantic target
    initializer_decision.json
    initializer_decision_validation.json
    falsification_plan.json
    falsification_evidence.json
    falsification_review.json
    falsification_validation.json
    part_completion.json
    rev-NNN/face_labels.u32le
    rev-NNN/edit_manifest.json
    rev-NNN/renders/                      diagnostic, flat, and neutral views
    rev-NNN/selected-only/renders/VIEW.png isolated renders of this part alone
  final/                                  the request's required artifacts
  ```

  {part_directory_contract} Record the exact semantic name in each artifact's
  `semantic_part` field. Write `rev-NNN/face_labels.u32le` and
  `rev-NNN/edit_manifest.json` at exactly that depth, never nested deeper.
- `PART/rev-NNN/selected-only/renders/` must contain renders of that part *by
  itself*, with every other segment hidden, one file per camera named for its
  view. A whole-object render with the target merely tinted does not belong
  there: anyone auditing that directory expects the part in isolation, and a
  tinted whole-object image read as an isolated one hides both missing and
  extra geometry.
- Export the current candidate and build its isolated selected-only USD under
  `PART/rev-NNN/selected-only/`, then render that exact stage into
  `PART/rev-NNN/selected-only/renders/`. The review render manifest's `scene`
  must resolve to `PART/rev-NNN/selected-only/part-only.usdc` beside it so the
  launcher can checkpoint the exact reviewed revision without overwriting an
  earlier lock's evidence.
- Keep each part's evidence under its own `PART` directory. Do not invent
  parallel per-part directories such as `candidate-vN`, `inspect-*`, or
  `review-*` variants, and do not place run files outside the run directory.
- Prepare the triangular USD and generate one immutable deterministic fragment
  map with `scripts/oversegment_mesh.py` unless the frozen request contains a
  continuation seed. A continuation must reuse its staged fragment map
  unchanged. Every semantic include/exclude operation applies to whole
  fragment IDs.
- Before planning or selecting fragments, render the neutral source and frozen
  fragment stage from exactly these eight cube corners: `+x-y+z`, `-x-y+z`,
  `+x+y+z`, `-x+y+z`, `+x-y-z`, `-x-y-z`, `+x+y-z`, and `-x+y-z`. Reuse all
  eight saved cameras for registered ID evidence and final held-out review.
  Memory never replaces or reduces this evidence set.
- Bind the neutral render manifest, zero-label ID buffers, and topology
  component builder to the exact immutable input USD, not the prepared neutral
  derivative. Read camera JSON paths from the render manifest's
  `.renders[].camera`; never glob response JSON files as cameras.
{image_initializer_contract}
- Inspect the registered masks, surviving pixels, and projected whole
  fragments together when image evidence is available. Record and validate
  `PART/initializer_decision.json`. Run
  `scripts/validate_initializer_decision.py` to create
  `PART/initializer_decision_validation.json` before recording initializer
  memory or creating rev-000. `PART/initializer_runtime_gate.json` is
  launcher-owned watchdog state; it never substitutes for decision validation
  and must not be cited as if it did. After validation passes, do not rewrite
  the decision or any cited evidence. If one changes, delete the stale
  validation, rerun the validator, and wait for the runtime gate to accept the
  refreshed corpus before continuing. Do not start or review rev-000 against a
  stale digest. Part size or category alone must not choose the route.
- If the decision is `image_mask_seed`, reuse the three accepted probe views,
  generate the other five cube-corner views, and use
  `scripts/build_deterministic_fragment_union.py` for the exact eight-view set
  union. Never use negative pixels, ratios, voting, vetoes, or adjacency growth.
  Apply that edit unchanged as rev-000 and verify byte identity.
- If the decision is `direct_agentic_selection`, discard every mask-derived
  fragment ID. Add focused neutral/ID views, enumerate the logical surfaces,
  batch closest-hit positive and nearby negative fragment picks, inspect each
  whole fragment, and build rev-000 only from confirmed direct selections.
- After the eight-view neutral and zero-label ID evidence exists, run
  `scripts/build_topology_component_evidence.py` once with
  `--appearance-dir inputs/references`. Its visible cards pair registered source
  appearance with a wider local crop and a readable full-view locator. A card's
  nearby-component list only proposes similarly scaled 3D neighbors to inspect;
  it does not prove shared semantics. The gallery's isolated cards for components
  hidden in every assembled OVRTX view include a diagnostic-only local preview
  and global-position locator. Never use that PIL preview for a semantic or
  inspectability decision; obtain a focused shared OVRTX render first, or report
  the component uninspectable.
- Use that corpus only while reviewing the active part. Inspect likely cards and
  named confusers. Do not make a global
  component-to-semantics assignment before part work; an early visual guess
  must not become a sticky taxonomy through memory.
- Before the first semantic revision, use the component corpus to challenge
  the plan's assembly granularity. Topology is a clue, not semantic proof.
  Prefer the coarsest logical-assembly partition consistent with the request
  and registered evidence; a semantic part is not every surface that can be
  named. When the request supplies names but no annotated boundary or explicit
  part description, do not infer a fine cut inside a welded component from the
  name alone. Surface orientation, a narrow visible band, and even a local seam
  are supporting clues, not sufficient evidence. Split only when registered
  multi-view evidence establishes a repeatable closed boundary around a
  coherent 3D subassembly, or the user explicitly defines that boundary.
  Otherwise revise the still-unstarted plan at the logical-assembly level and
  keep the component together.
- State history is append-only. Never rewrite an existing
  `state/labels-NNN.u32le`, especially one sealed by an initializer gate. For a
  correction, use the newest state as the revision parent, write a new part
  revision, and promote its labels to a new monotonically numbered state file.
  Do not rebuild or reapply unchanged later parts. A cross-label transfer uses
  two append-only revisions: exclude from the source, then include in the
  destination from the resulting newest state.
- Treat `state/final_labels.u32le` as a launcher-owned terminal promotion
  marker, not a rolling working alias. Never create, replace, rename, or write
  that path. During construction and final export preparation use the newest
  numbered state. After every requested part has a complete validated lock and
  the bounded uncertainty pass is complete, end the turn. The launcher validates
  the full lock corpus, promotes the exact lock-bound numbered state, and starts
  a continuation turn when final export work remains. In that continuation,
  read `raw/final_labels_promotion.json` and use its lock-bound `source_labels`
  numbered state for final export work; never access or change the terminal path.
- When memory is enabled, the launcher checkpoints the exact candidate as soon
  as deterministic selected-only and frontier evidence exists. On a deliberate
  revisit, search that checkpoint before gathering new discriminating evidence,
  then record any changed semantic diagnosis or correction. The new evidence,
  not the memory summary, decides the correction. The launcher checkpoint does
  not satisfy your agent-origin required capture; record a changed diagnosis,
  correction, deferral, or validated lock yourself through the broker.
- For rev-000, pass `PART/rev-000` itself as the output directory to
  `apply_fragment_edits.py`; its generated `face_labels.u32le`,
  `diagnostic.usdc`, and `edit_manifest.json` must all remain there. Never use a
  `rev-000-applied` output and copy only selected files afterward. Do not
  pre-create the revision directory; the edit helper owns it and requires it
  not to exist.
- Never blend mask-derived and direct fragment IDs in one rev-000.
- Treat every rev-000 as provisional, including an accepted image-mask seed.
  Before reviewing it, write a falsification plan with expected instances,
  named confusers, observable falsifiers, construction views, and held-out
  review views.
- Refine by direct visual fragment decisions. Compare the neutral render, flat
  binary label view, fragment colors, face-ID buffer, and fragment-ID buffer
  from exactly matching saved cameras. Batch polygons, scribbles, or clicks and
  map them through the closest-visible ID buffers.
- Every review must separately search for false positives and false negatives.
  Lit images provide semantic context only; flat label and ID buffers provide
  label truth and mesh correspondence.
- Before acceptance, export the exact candidate, hide every segment except the
  active part with `scripts/build_selected_only_stage.py`, and render that
  selected-only stage from every held-out view. For repeated instances, compare
  silhouettes, shells, openings, and internal surfaces against one another.
  Use these renders for both error directions: reject extra selected geometry,
  but also reject truncated shells, discontinuous contours, implausible
  openings, and unexplained jagged cuts caused by missing faces.
- Candidate exports must configure exactly the segment IDs present in that
  revision's labels. Do not include future empty labels or residual label `0`
  after it has no faces.
- Connected-component count proves instance presence, not completeness. Record
  `instance_completeness_reviews` for every expected instance. Inspect at least
  two visible selected-only views per instance, spanning `primary_surface` and
  `opposing_or_occluded_surface`. Classify every visible selection boundary as
  an intended semantic boundary or silhouette/occlusion; any unexplained
  boundary rejects the candidate.
- Expand an explicit count or repeated collection into separate stable
  `expected_instances` IDs before review (for example, `wing-1` through
  `wing-4`, never one entry called `four wings`). The number of
  `instance_completeness_reviews` entries must match that explicit count; a
  collective phrase cannot stand in for its instances. If any instance is
  missing or unresolved, inspect the selected components'
  `nearby_similar_scale_component_ids` and record each hint's disposition.
  These hints are candidates to falsify, not semantic proof. An instance-count
  mismatch cannot pass: revise when evidence localizes it, otherwise defer.
- In every completeness view, inspect at least one actual adjacent unselected
  frontier fragment in matching flat-label and ID evidence. If it belongs to
  the target, include the whole fragment and rerender. The completion review
  must bind those fragment IDs to the exact final frontier audit.
- Independently collect signed positive target clicks and signed negative
  confuser clicks from focused evidence. Do not use easy center-only samples.
  For every expected instance, cover target interior, visible extent, and an
  opposing or occluded surface from at least two views, and place a close
  positive/negative pair across a visible semantic boundary. Tag negative
  clicks with the exact planned confuser, and collect both polarities in every
  held-out view. Run `compare_face_labels.py` against the exact current
  candidate. A selected negative face or omitted positive face rejects the
  candidate regardless of the agent's visual confidence.
- Do not lock until `PART/falsification_review.json` describes the latest
  revision and `scripts/validate_falsification_review.py` writes a passing
  `PART/falsification_validation.json`. Preserve at least one held-out review
  view not used to construct or last edit the candidate. Never hand-write the
  validation file and never edit the review or bound evidence after validation;
  a change requires deleting and rerunning the validation.
- Retry while a changed camera, evidence source, or edit still produces
  measurable progress: it covers a previously uncovered required positive,
  removes at least one signed mismatch, resolves an unexplained boundary, or
  adds a registered view that can decide a named ambiguity. Defer only after
  distinct changed attempts repeat the same defect without any of those gains.
  Preserve the best valid candidate and record the unresolved issue in memory.
- Make revisions evidence-monotonic. Replace the current best candidate only
  when new signed, registered, selected-only, or component-context evidence
  identifies a specific false positive or false negative and the edit addresses
  it. An ambiguous reinterpretation is not new evidence: keep the earlier
  candidate unchanged, gather a genuinely discriminating view, or defer. Never
  swap one plausible semantic guess for another merely to try something different.
- When deferring, defer the part and continue with another part. Revisit the
  deferred part once after every other non-deferred requested part has been
  attempted.
- Do not defer an unstarted part. For every requested part with inspectable
  evidence, create and visually review at least one real rev-000 covering every
  inspectable expected instance and visible extent. A token seed does not count.
  Complexity, remaining time, or strict validation cost alone is not a deferral
  reason.
- Treat every requested target as a positive assertion that the part should be
  found. Before calling one absent or uninspectable, audit every still-unlocked
  topology component, not only likely-looking components. Tiny assembled-view
  crops and semantic shape guesses cannot rule a component out. For each hidden,
  occluded, or too-small component that remains unclassified, render isolated
  geometry from at least two fitted opposing views and compare it with registered
  source context. Record every component's disposition; defer only when this
  exhaustive residual audit finds no consistent component or component assembly.
- Only validated locks reserve faces. A deferred or provisional candidate must
  not consume geometry from later parts; preserve it separately and rebuild or
  reconcile it from the latest locked parent when revisited.
{locked_revision_contract}
- Treat a zero unselected frontier as not applicable when the target is already
  a complete disconnected component. Do not invent adjacent geometry to make a
  checklist pass; leave the part provisional if validation cannot express it.
- Never rewrite the frozen fragment artifacts. If their integrity changes,
  stop semantic edits and reproduce the deterministic map from the source once.
  Stop only if that clean retry cannot restore provenance or label integrity.
- Do not use annulus fitting, PCA, flood filling, category-specific geometry
  predicates, remembered IDs, fixed coordinates, or asset-specific scripts.
- When a part locks, write `PART/part_completion.json` with exactly these
  fields: `schema_version` set to `mesh-segmentation-part-completion.v1`;
  `status` set to `locked`; `semantic_part` set to the exact target name;
  `segment_id`; `final_revision` equal to the `revision` its
  `falsification_review.json` describes; `falsification_review` set to the
  literal string `falsification_review.json`; `falsification_validation` set
  to the literal string `falsification_validation.json`; and
  `falsification_validation_sha256` set to the SHA-256 of that validation
  file, computed after the file is final. A record missing the schema version,
  the `locked` status, or a current digest does not lock the part.
- When memory is enabled, record the agent-authored validated-lock observation
  immediately after `part_completion.json`. Do this for every requested part
  before ending the lock-completion turn; the launcher's first terminal
  promotion closes the timely-memory window.
- Evidence acquisition may be batched, but initializer decisions, rev-000
  creation, refinement, and locking are sequential within the active part. Do
  not promote a deferred candidate as a validated lock.
- Process broad residual body/other categories last. Export one mesh prim per
  segment with exact source-face provenance and preserve every prompt, raw
  overlay, affine matrix, mask, ID buffer, region, edit, render, and decision.
- The root `segments.json` is the manifest for the complete final partition,
  unlike each revision-local candidate manifest. Before promotion, make its
  configured segment IDs exactly match the IDs present in the newest numbered
  state; after launcher promotion, verify they exactly match
  `state/final_labels.u32le`. Include residual label 0 only when it has faces,
  and prune superseded or emptied IDs.
- Residual is not a shortcut. Before export, revisit any visible named geometry
  still absorbed by body or `other`, especially repeated or occluded components.
- Before export, make one bounded uncertainty pass over at most three weak
  candidates. Prefer an instance-count mismatch, a multi-piece assembly with
  nearby peers left in another segment or the residual, or an unresolved named
  confuser. In a memory-enabled run, search the exact part checkpoint and
  inspect its candidate labels and frontier artifact before gathering new
  evidence or creating a revisit revision; the memory search must be the first
  action of the revisit. Otherwise read its local candidate artifacts. For a
  cross-label removal or transfer, search both affected parts first and require
  a source-context view showing the component's assembled attachment or
  appearance; selected-only geometry alone is not enough. Create at most one
  new revision per revisited part in this pass, and only if the comparison
  localizes an error; otherwise keep it and move on.
- Never implement a revisit by rewriting the original sequential
  `state/labels-*` chain; append only the changed part revisions to the newest
  state.
- Render the exported USD from held-out views before terminal acceptance.
- Write `final/target_evidence.json` as
  `{{"targets": [...]}}` with exactly one record per requested semantic target,
  in the same order as the frozen request's `inputs.target_semantic_parts`,
  no duplicates and no extra records. Give each record its exact
  `target_semantic_part` name, `status: "locked"`, and a
  `falsification_validation` path pointing at that part's passing
  `PART/falsification_validation.json`. Write that path relative to the run
  directory (`PART/falsification_validation.json`), not relative to `final/`.
- Always export the best complete face partition, even when some parts remain
  deferred: preserve one label per source face and produce the final USD,
  labels, manifest, renders, and report. Clearly distinguish provisional parts
  from validated locks.
- Produce every path listed in the frozen request's
  `required_final_artifacts` array at exactly that run-relative location, except
  that only the launcher may write `state/final_labels.u32le`.

Finish only after every requested artifact is internally consistent. Report a
fragment-boundary limitation or uninspectable region instead of silently
accepting a visible defect.
""",
            helper_scripts,
        )
    return _bind_mesh_helper_paths(
        f"""Use the `content-workflow-mesh-segmentation` and
`usd-cli` skills to complete a fresh, blind mesh-segmentation run.

Frozen request: `{request_path}`
Run directory and complete permitted evidence corpus: `{run_dir}`

Read the frozen request and every staged reference image before acting.
{target_mode}

Execution contract:
- Work natively; never start Docker and never use image generation.
- Do not create, project, or vote over 2D semantic masks.
- Do not inspect parent directories, sibling runs, git history, golden labels,
  prior experiments, or prior run summaries.
- Prepare the immutable triangular USD with `scripts/prepare_mesh.py`, then run
  deterministic over-segmentation with `scripts/oversegment_mesh.py`. Freeze
  that fragment map for the whole run.
- Observe neutral multi-view renders, then batch closest-hit positive and
  negative picks for the active semantic part. Every ray hit records an exact
  source face for provenance and immediately resolves to one immutable
  fragment.
- Write a hypothesis card, fit the simplest replayable geometric model that
  explains the signed evidence, atomize its face-space proposal to whole
  fragments, and preserve every failed hypothesis.
- Render registered color, normal, and linear-depth evidence. Add opposing,
  axial, grazing, endpoint, underside, selected-only, or fragment-color views only
  when they answer a concrete ambiguity.
- Diagnose coherent errors before editing. After two revisions with the same
  failure pattern, change the model family, evidence, instance grouping, or
  view instead of oscillating over boundary fragments.
- Never assign individual triangles. Exact corrections include or exclude
  whole fragment IDs. If a fragment visibly spans two semantic parts, report
  inadequate over-segmentation instead of splitting it with face edits.
- Lock one accepted dense-label revision before opening the next part. Later
  parts may assign only currently unclassified fragments.
- Process the broad residual body last, record residual decisions, and require
  exactly one label per source face.
- Export one mesh prim per segment with exact source-face provenance. Render
  the exported USD from held-out views and produce every path listed in
  `required_final_artifacts`.
- Preserve the fragment manifest/map, all hypotheses, clicks, analyses,
  cameras, renders, comparisons, frontier audits, and acceptance decisions
  under the run directory.

Finish only after the exported USD, dense labels, manifest, held-out renders,
and terminal report are internally consistent. Report unresolved uncertainty
instead of silently accepting a visible defect.
""",
        helper_scripts,
    )


def _validate_terminal_artifacts(
    run_dir: Path,
    *,
    required_artifacts: list[str] | tuple[str, ...] = REQUIRED_FINAL_ARTIFACTS,
    authoritative_request: dict[str, object] | None = None,
    part_locks_only: bool = False,
    include_part_queue: bool = True,
) -> dict[str, object]:
    validate_targeted_lock_bindings = (
        part_locks_only or "final/target_evidence.json" in required_artifacts
    )
    missing = [
        relative
        for relative in required_artifacts
        if not (run_dir / relative).is_file()
    ]
    semantic_errors: list[str] = []
    if authoritative_request is None:
        request, request_error = _load_json_object(
            run_dir / "request.json",
            label="request.json",
        )
    else:
        request = authoritative_request
        request_error = None
    if request_error:
        semantic_errors.append(request_error)
    elif request is not None and (
        "content-workflow-mesh-segmentation" in request.get("required_skills", [])
    ):
        expected_initializer_parts: list[str] = []
        inputs = request.get("inputs")
        targets = (
            inputs.get("target_semantic_parts", []) if isinstance(inputs, dict) else []
        )
        if not isinstance(targets, list):
            semantic_errors.append("request.json target_semantic_parts must be a list")
        else:
            terminal_parts = list(targets)
            queue_path = run_dir / "part_work_queue.json"
            if include_part_queue and queue_path.is_file():
                queue, queue_error = _load_json_object(
                    queue_path,
                    label="part_work_queue.json",
                )
                if queue_error:
                    semantic_errors.append(queue_error)
                elif queue is not None and isinstance(queue.get("parts"), list):
                    for part in queue["parts"]:
                        if not isinstance(part, dict):
                            continue
                        part_name = part.get("name", part.get("part_name"))
                        name_error = _semantic_part_name_error(part_name)
                        if name_error:
                            semantic_errors.append(
                                f"Queue part name {part_name!r} {name_error}"
                            )
                        elif part_name not in terminal_parts:
                            assert isinstance(part_name, str)
                            terminal_parts.append(part_name)
            for target in terminal_parts:
                name_error = _semantic_part_name_error(target)
                if name_error:
                    semantic_errors.append(
                        f"Target semantic part {target!r} {name_error}"
                    )
                    continue
                assert isinstance(target, str)
                expected_initializer_parts.append(target)
                # The directory only has to fold to the target name; resolve it
                # so a slugged or `parts/`-nested layout is still validated.
                target_dir = _resolve_part_dir(run_dir, target)
                part_prefix = (
                    str(target_dir.relative_to(run_dir.resolve()))
                    if target_dir is not None
                    else target
                )
                decision_relative = f"{part_prefix}/initializer_decision.json"
                validation_relative = (
                    f"{part_prefix}/initializer_decision_validation.json"
                )
                review_relative = f"{part_prefix}/falsification_review.json"
                falsification_relative = f"{part_prefix}/falsification_validation.json"
                completion_relative = f"{part_prefix}/part_completion.json"
                decision, decision_error = _load_json_object(
                    run_dir / decision_relative,
                    label=decision_relative,
                )
                validation, validation_error = _load_json_object(
                    run_dir / validation_relative,
                    label=validation_relative,
                )
                review, review_error = _load_json_object(
                    run_dir / review_relative,
                    label=review_relative,
                )
                falsification, falsification_error = _load_json_object(
                    run_dir / falsification_relative,
                    label=falsification_relative,
                )
                completion, completion_error = _load_json_object(
                    run_dir / completion_relative,
                    label=completion_relative,
                )
                if decision_error:
                    semantic_errors.append(decision_error)
                    continue
                if validation_error:
                    semantic_errors.append(validation_error)
                    continue
                if review_error:
                    semantic_errors.append(review_error)
                    continue
                if falsification_error:
                    semantic_errors.append(falsification_error)
                    continue
                if completion_error:
                    semantic_errors.append(completion_error)
                    continue
                if (
                    decision is None
                    or validation is None
                    or review is None
                    or falsification is None
                    or completion is None
                ):
                    continue
                if not _schema_matches(
                    decision.get("schema_version"),
                    "mesh-segmentation-initializer-decision.v1",
                ):
                    semantic_errors.append(
                        f"{decision_relative} has an unsupported schema"
                    )
                if decision.get("status") != "accepted":
                    semantic_errors.append(
                        f"{decision_relative} status must be 'accepted'"
                    )
                if decision.get("decision") not in {
                    "image_mask_seed",
                    "direct_agentic_selection",
                }:
                    semantic_errors.append(
                        f"{decision_relative} has an unsupported decision"
                    )
                if not _schema_matches(
                    validation.get("schema_version"),
                    "mesh-segmentation-initializer-decision-validation.v1",
                ):
                    semantic_errors.append(
                        f"{validation_relative} has an unsupported schema"
                    )
                if validation.get("status") != "passed":
                    semantic_errors.append(
                        f"{validation_relative} status must be 'passed'"
                    )
                if validation.get("semantic_part") != target:
                    semantic_errors.append(
                        f"{validation_relative} semantic_part does not match "
                        f"target {target!r}"
                    )
                if validation.get("decision") != decision.get("decision"):
                    semantic_errors.append(
                        f"{validation_relative} does not validate the recorded "
                        "initializer decision"
                    )
                if validation.get("decision_sha256") != _sha256_file(
                    run_dir / decision_relative
                ):
                    semantic_errors.append(
                        f"{validation_relative} decision digest is stale"
                    )
                if not _schema_matches(
                    review.get("schema_version"),
                    "mesh-segmentation-falsification-review.v1",
                ):
                    semantic_errors.append(
                        f"{review_relative} has an unsupported schema"
                    )
                if review.get("status") != "accepted":
                    semantic_errors.append(
                        f"{review_relative} status must be 'accepted'"
                    )
                if review.get("semantic_part") != target:
                    semantic_errors.append(
                        f"{review_relative} semantic_part does not match "
                        f"target {target!r}"
                    )
                if review.get("false_positive_review") != "passed":
                    semantic_errors.append(
                        f"{review_relative} false-positive review did not pass"
                    )
                if review.get("false_negative_review") != "passed":
                    semantic_errors.append(
                        f"{review_relative} false-negative review did not pass"
                    )
                if review.get("actionable_issues") != []:
                    semantic_errors.append(
                        f"{review_relative} still records actionable issues"
                    )
                if not _schema_matches(
                    falsification.get("schema_version"),
                    "mesh-segmentation-falsification-validation.v1",
                ):
                    semantic_errors.append(
                        f"{falsification_relative} has an unsupported schema"
                    )
                if falsification.get("status") != "passed":
                    semantic_errors.append(
                        f"{falsification_relative} status must be 'passed'"
                    )
                if falsification.get("semantic_part") != target:
                    semantic_errors.append(
                        f"{falsification_relative} semantic_part does not match "
                        f"target {target!r}"
                    )
                if falsification.get("review_sha256") != _sha256_file(
                    run_dir / review_relative
                ):
                    semantic_errors.append(
                        f"{falsification_relative} review digest is stale"
                    )
                if validate_targeted_lock_bindings:
                    if falsification.get("revision") != review.get("revision"):
                        semantic_errors.append(
                            f"{falsification_relative} revision does not match review"
                        )
                    candidate_path = _manifest_run_artifact_path(
                        run_dir,
                        falsification.get("candidate_labels"),
                    )
                    if candidate_path is None or not candidate_path.is_file():
                        semantic_errors.append(
                            f"{falsification_relative} candidate labels are "
                            "missing or unsafe"
                        )
                    elif falsification.get("candidate_labels_sha256") != _sha256_file(
                        candidate_path
                    ):
                        semantic_errors.append(
                            f"{falsification_relative} candidate-label digest is stale"
                        )
                if completion.get("schema_version") != (
                    "mesh-segmentation-part-completion.v1"
                ):
                    semantic_errors.append(
                        f"{completion_relative} has an unsupported schema"
                    )
                if completion.get("status") != "locked":
                    semantic_errors.append(
                        f"{completion_relative} status must be 'locked'"
                    )
                if completion.get("semantic_part") != target:
                    semantic_errors.append(
                        f"{completion_relative} semantic_part does not match "
                        f"target {target!r}"
                    )
                if completion.get("final_revision") != review.get("revision"):
                    semantic_errors.append(
                        f"{completion_relative} does not name the falsified "
                        "final revision"
                    )
                if completion.get("falsification_review") != (
                    "falsification_review.json"
                ):
                    semantic_errors.append(
                        f"{completion_relative} must record falsification_review.json"
                    )
                if completion.get("falsification_validation") != (
                    "falsification_validation.json"
                ):
                    semantic_errors.append(
                        f"{completion_relative} must record "
                        "falsification_validation.json"
                    )
                if completion.get("falsification_validation_sha256") != (
                    _sha256_file(run_dir / falsification_relative)
                ):
                    semantic_errors.append(
                        f"{completion_relative} falsification digest is stale"
                    )
        semantic_errors.extend(
            _validate_initializer_runtime_gates(
                run_dir,
                expected_parts=expected_initializer_parts,
            )
        )
    if part_locks_only:
        return {
            "schema_version": TERMINAL_VALIDATION_SCHEMA_VERSION,
            "checked_at": utc_now(),
            "valid": not missing and not semantic_errors,
            "required_artifacts": list(required_artifacts),
            "missing_artifacts": missing,
            "semantic_validation_errors": semantic_errors,
            "agent_memory": {"enabled": False, "status": "not_checked"},
            "agent_memory_warnings": [],
        }
    targeted = "final/target_evidence.json" in required_artifacts
    recognition = "part_lock_manifest.json" in required_artifacts
    if recognition and not missing:
        semantic_errors.extend(validate_recognition_part_locks(run_dir))
    if targeted and not missing:
        semantic_errors.extend(
            validate_targeted_selection_evidence(
                run_dir,
                authoritative_request=request,
            )
        )
    export_relative = "final/export_manifest.json"
    if export_relative in required_artifacts and export_relative not in missing:
        export_manifest, export_error = _load_json_object(
            run_dir / export_relative,
            label=export_relative,
        )
        if export_error:
            semantic_errors.append(export_error)
        elif export_manifest is not None:
            if export_manifest.get("status") != "passed":
                semantic_errors.append("Export manifest status must be 'passed'")
            if export_manifest.get("exact_source_face_coverage") is not True:
                semantic_errors.append(
                    "Export manifest must confirm exact source-face coverage"
                )
            if export_manifest.get("semantic_decision_unit") != "immutable_fragment":
                semantic_errors.append(
                    "Export manifest must declare immutable-fragment decisions"
                )
            output_usd_relative = "final/segmented.usdc"
            if (
                output_usd_relative in required_artifacts
                and output_usd_relative not in missing
                and export_manifest.get("output_usd_sha256")
                != _sha256_file(run_dir / output_usd_relative)
            ):
                semantic_errors.append(
                    "Export manifest output USD digest must match final/segmented.usdc"
                )
            state_labels_path = run_dir / "state" / "final_labels.u32le"
            if state_labels_path.is_file():
                state_labels_sha256 = _sha256_file(state_labels_path)
                if export_manifest.get("face_labels_sha256") != state_labels_sha256:
                    semantic_errors.append(
                        "Export manifest face-label digest must match "
                        "state/final_labels.u32le"
                    )
                final_labels_path = run_dir / "final" / "face_labels.u32le"
                if (
                    final_labels_path.is_file()
                    and _sha256_file(final_labels_path) != state_labels_sha256
                ):
                    semantic_errors.append(
                        "Final face labels must exactly match state/final_labels.u32le"
                    )
    fragment_labels_relative = "fragments/fragment_ids.u32le"
    final_labels_relative = "state/final_labels.u32le"
    if (
        fragment_labels_relative in required_artifacts
        and final_labels_relative in required_artifacts
        and fragment_labels_relative not in missing
        and final_labels_relative not in missing
    ):
        fragment_labels, fragment_error = _load_u32_labels(
            run_dir / fragment_labels_relative
        )
        final_labels, final_error = _load_u32_labels(run_dir / final_labels_relative)
        if fragment_error:
            semantic_errors.append(fragment_error)
        if final_error:
            semantic_errors.append(final_error)
        if fragment_labels is not None and final_labels is not None:
            if len(fragment_labels) != len(final_labels):
                semantic_errors.append(
                    "Fragment and final semantic label counts must match"
                )
            else:
                semantic_by_fragment: dict[int, int] = {}
                split_fragments: set[int] = set()
                for fragment_id, semantic_id in zip(
                    fragment_labels,
                    final_labels,
                    strict=True,
                ):
                    prior = semantic_by_fragment.setdefault(fragment_id, semantic_id)
                    if prior != semantic_id:
                        split_fragments.add(fragment_id)
                if split_fragments:
                    semantic_errors.append(
                        "Final labels split immutable fragments: "
                        f"{sorted(split_fragments)[:32]}"
                    )
    continuation_manifest_path = run_dir / "continuation_seed" / "manifest.json"
    if continuation_manifest_path.is_file():
        semantic_errors.extend(_validate_continuation_seed_locks(run_dir))
    memory_validation, memory_warnings = _validate_agent_memory(request)
    return {
        "schema_version": TERMINAL_VALIDATION_SCHEMA_VERSION,
        "checked_at": utc_now(),
        "valid": not missing and not semantic_errors,
        "required_artifacts": list(required_artifacts),
        "missing_artifacts": missing,
        "semantic_validation_errors": semantic_errors,
        "agent_memory": memory_validation,
        "agent_memory_warnings": memory_warnings,
    }


def _validate_agent_memory(
    request: dict[str, object] | None,
) -> tuple[dict[str, object], list[str]]:
    """Verify that an enabled mesh run durably recorded mesh observations."""

    runtime = request.get("runtime") if isinstance(request, dict) else None
    memory_config = runtime.get("agent_memory") if isinstance(runtime, dict) else None
    if not isinstance(memory_config, dict) or memory_config.get("enabled") is not True:
        return {"enabled": False, "status": "disabled", "observation_count": 0}, []

    errors: list[str] = []
    run_id = memory_config.get("run_id")
    root_value = memory_config.get("root")
    if not isinstance(run_id, str) or not run_id:
        errors.append("Enabled agent memory lacks a run_id")
    if not isinstance(root_value, str) or not root_value:
        errors.append("Enabled agent memory lacks a root")
    if errors:
        return {
            "enabled": True,
            "status": "failed",
            "observation_count": 0,
        }, errors

    assert isinstance(run_id, str)
    assert isinstance(root_value, str)
    root = Path(root_value)
    manifest_path = root / run_id / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"Agent memory manifest is missing: {manifest_path}")
        return {
            "enabled": True,
            "status": "failed",
            "run_id": run_id,
            "root": str(root),
            "observation_count": 0,
        }, errors

    try:
        memory = AgentMemory(run_id=run_id, memory_root=root)
        observation_count = memory.count_observations(workflow="mesh-segmentation")
        require_origin = request.get("schema_version") == REQUEST_SCHEMA_VERSION
        agent_observation_count = (
            memory.count_observations(
                workflow="mesh-segmentation",
                tag=MEMORY_ORIGIN_AGENT_TAG,
            )
            if require_origin
            else observation_count
        )
        launcher_observation_count = (
            memory.count_observations(
                workflow="mesh-segmentation",
                tag=MEMORY_ORIGIN_LAUNCHER_TAG,
            )
            if require_origin
            else 0
        )
        context = memory.context(limit=12)
    except Exception as exc:  # noqa: BLE001 - terminal contract reports corruption
        errors.append(
            f"Could not validate required agent memory: {type(exc).__name__}: {exc}"
        )
        return {
            "enabled": True,
            "status": "failed",
            "run_id": run_id,
            "root": str(root),
            "observation_count": 0,
        }, errors

    if observation_count == 0:
        errors.append(
            "Required agent memory contains no mesh-segmentation observations"
        )
    elif agent_observation_count == 0:
        errors.append(
            "Required agent memory contains no agent-authored observation; "
            "launcher checkpoints alone do not satisfy capture"
        )
    timely_target_observation_ids: dict[str, list[str]] = {}
    inputs = request.get("inputs") if isinstance(request, dict) else None
    raw_targets = (
        inputs.get("target_semantic_parts") if isinstance(inputs, dict) else None
    )
    isolation = request.get("isolation") if isinstance(request, dict) else None
    working_directory = (
        isolation.get("working_directory") if isinstance(isolation, dict) else None
    )
    final_labels_path = (
        Path(working_directory) / "state" / "final_labels.u32le"
        if isinstance(working_directory, str) and working_directory
        else None
    )
    target_parts: list[str] = []
    if isinstance(raw_targets, list):
        for target in raw_targets:
            if _semantic_part_name_error(target) is None:
                assert isinstance(target, str)
                target_parts.append(target)
    if isinstance(working_directory, str) and working_directory:
        queue_path = Path(working_directory) / "part_work_queue.json"
        if queue_path.is_file():
            queue, queue_error = _load_json_object(
                queue_path,
                label="part_work_queue.json",
            )
            if queue_error:
                errors.append(queue_error)
            elif queue is not None and isinstance(queue.get("parts"), list):
                for part in queue["parts"]:
                    if not isinstance(part, dict):
                        continue
                    part_name = part.get("name", part.get("part_name"))
                    if _semantic_part_name_error(part_name) is None:
                        assert isinstance(part_name, str)
                        target_parts.append(part_name)
    target_parts = list(dict.fromkeys(target_parts))
    if target_parts and final_labels_path is not None and final_labels_path.is_file():
        promotion, promotion_error = _load_json_object(
            final_labels_path.parents[1] / "raw" / "final_labels_promotion.json",
            label="raw/final_labels_promotion.json",
        )
        final_labels_time: float | None = None
        if request.get("schema_version") == REQUEST_SCHEMA_VERSION:
            if promotion_error or promotion is None:
                errors.append(
                    promotion_error or "Final-label promotion marker is missing"
                )
            elif promotion.get("schema_version") != (
                FINAL_LABELS_PROMOTION_SCHEMA_VERSION
            ):
                errors.append("Final-label promotion marker schema is unsupported")
            elif promotion.get("final_labels_sha256") != _sha256_file(
                final_labels_path
            ):
                errors.append("Final-label promotion marker digest is stale")
            else:
                observed_at = promotion.get("observed_at")
                try:
                    if not isinstance(observed_at, str):
                        raise ValueError("observed_at must be a string")
                    final_labels_time = datetime.fromisoformat(observed_at).timestamp()
                except ValueError as exc:
                    errors.append(
                        f"Final-label promotion marker time is invalid: {exc}"
                    )
        else:
            # Pre-v3 run bundles predate the launcher-owned first-seen marker.
            final_labels_time = final_labels_path.stat().st_mtime
        for target in target_parts:
            if final_labels_time is None:
                continue
            try:
                timely_cards = memory.search(
                    MemorySearchQuery(
                        workflow="mesh-segmentation",
                        target=target,
                        created_before=datetime.fromtimestamp(
                            final_labels_time + MEMORY_PROMOTION_GRACE_SECONDS,
                            tz=UTC,
                        ).isoformat(),
                        tag=(MEMORY_ORIGIN_AGENT_TAG if require_origin else None),
                        limit=1,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - memory remains advisory
                errors.append(
                    "Could not validate agent-memory timing for target part "
                    f"{target!r}: {type(exc).__name__}: {exc}"
                )
                continue
            if timely_cards:
                timely_target_observation_ids[target] = [timely_cards[0].observation_id]
            else:
                errors.append(
                    "Required agent memory lacks an observation recorded before "
                    f"final label promotion for target part {target!r}"
                )
    memory_passed = bool(agent_observation_count and not errors)
    return {
        "enabled": True,
        "status": "passed" if memory_passed else "warning",
        "run_id": run_id,
        "root": str(root),
        "observation_count": observation_count,
        "agent_observation_count": agent_observation_count,
        "launcher_observation_count": launcher_observation_count,
        # Kept for old dashboard readers; phases are intentionally not used as
        # evidence of who captured an observation.
        "operational_observation_count": agent_observation_count,
        "timely_target_observation_ids": timely_target_observation_ids,
        "bounded_context_observation_ids": [
            card.observation_id for card in context.cards
        ],
        "unresolved_observation_ids": list(context.unresolved_observation_ids),
        "pinned_observation_ids": list(context.pinned_observation_ids),
    }, errors


def _validate_continuation_seed_locks(run_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest, manifest_error = _load_json_object(
        run_dir / "continuation_seed" / "manifest.json",
        label="continuation_seed/manifest.json",
    )
    if manifest_error:
        return [manifest_error]
    assert manifest is not None
    if manifest.get("schema_version") != (
        "content-agents.mesh-segmentation-continuation-seed.v1"
    ):
        errors.append("Continuation seed manifest has an unsupported schema")
    fragment_path = run_dir / "fragments" / "fragment_ids.u32le"
    expected_fragment_digest = manifest.get("fragment_labels_sha256")
    if not fragment_path.is_file() or expected_fragment_digest != _sha256_file(
        fragment_path
    ):
        errors.append("Continuation fragment map changed after staging")
    immutable_path = run_dir / "continuation_seed" / "immutable_face_labels.u32le"
    final_path = run_dir / "state" / "final_labels.u32le"
    immutable, immutable_error = _load_u32_labels(immutable_path)
    final, final_error = _load_u32_labels(final_path)
    if immutable_error:
        errors.append(immutable_error)
    if final_error:
        errors.append(final_error)
    if immutable is None or final is None:
        return errors
    if len(immutable) != len(final):
        errors.append("Continuation and final face-label counts differ")
        return errors
    background_segment_id = manifest.get("background_segment_id")
    if not isinstance(background_segment_id, int):
        errors.append("Continuation seed lacks an integer background segment ID")
        return errors
    changed_locked_faces = [
        index
        for index, (seed_value, final_value) in enumerate(
            zip(immutable, final, strict=True)
        )
        if seed_value != background_segment_id and final_value != seed_value
    ]
    if changed_locked_faces:
        errors.append(
            "Final labels changed continuation-locked faces: "
            f"{changed_locked_faces[:32]}"
        )
    return errors


def _load_json_object(
    path: Path, *, label: str
) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"Could not read {label}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{label} must be a JSON object"
    return payload, None


def _load_u32_labels(path: Path) -> tuple[array[int] | None, str | None]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return None, f"Could not read face labels {path}: {exc}"
    if len(payload) % 4:
        return None, f"Face-label artifact byte length is not divisible by four: {path}"
    labels: array[int] = array("I")
    labels.frombytes(payload)
    if sys.byteorder != "little":
        labels.byteswap()
    return labels, None


def validate_recognition_part_locks(run_dir: Path) -> list[str]:
    """Validate large-to-small, one-new-part-at-a-time recognition history."""

    errors: list[str] = []
    queue, queue_error = _load_json_object(
        run_dir / "part_work_queue.json",
        label="part_work_queue.json",
    )
    lock_manifest, lock_error = _load_json_object(
        run_dir / "part_lock_manifest.json",
        label="part_lock_manifest.json",
    )
    live_gate, live_gate_error = _load_json_object(
        run_dir / "live_sequential_gate.json",
        label="live_sequential_gate.json",
    )
    review_manifest, review_error = _load_json_object(
        run_dir / "part_review_manifest.json",
        label="part_review_manifest.json",
    )
    if queue_error:
        errors.append(queue_error)
    if lock_error:
        errors.append(lock_error)
    if live_gate_error:
        errors.append(live_gate_error)
    if review_error:
        errors.append(review_error)
    if queue is None or lock_manifest is None or live_gate is None:
        return errors
    if review_manifest is None:
        review_manifest = {}

    raw_parts = queue.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        return [*errors, "Recognition queue must contain at least one part"]
    parts = [part for part in raw_parts if isinstance(part, dict)]
    if len(parts) != len(raw_parts):
        errors.append("Every recognition queue entry must be a JSON object")
        return errors

    other_segment_id = queue.get("other_segment_id")
    if not isinstance(other_segment_id, int):
        errors.append("Recognition queue must reserve integer other_segment_id")
        return errors

    names: list[str] = []
    segment_ids: list[int] = []
    scope_ranks: list[int] = []
    body_indices: list[int] = []
    part_identities: list[tuple[str, int] | None] = []
    for index, part in enumerate(parts):
        name = part.get("name", part.get("part_name"))
        segment_id = part.get("segment_id")
        role = part.get("role")
        scope = part.get("estimated_scope")
        if not isinstance(name, str) or not name:
            errors.append(f"Queue entry {index} lacks a non-empty name")
            part_identities.append(None)
            continue
        name_error = _semantic_part_name_error(name)
        if name_error:
            errors.append(f"Queue part {name!r} {name_error}")
            part_identities.append(None)
            continue
        if not isinstance(segment_id, int) or segment_id == other_segment_id:
            errors.append(f"Queue part {name} has invalid segment_id")
            part_identities.append(None)
            continue
        part_identities.append((name, segment_id))
        if role not in {"distinctive", "body_fallback"}:
            errors.append(f"Queue part {name} has invalid role {role!r}")
        if scope not in RECOGNITION_SCOPE_ORDER:
            errors.append(f"Queue part {name} has invalid estimated_scope {scope!r}")
            scope_rank = -1
        else:
            scope_rank = RECOGNITION_SCOPE_ORDER[str(scope)]
        if role == "body_fallback":
            body_indices.append(index)
            if scope != "body":
                errors.append(f"Body fallback {name} must use estimated_scope 'body'")
        names.append(name)
        segment_ids.append(segment_id)
        scope_ranks.append(scope_rank)

    if len(set(names)) != len(names):
        errors.append("Recognition queue part names must be unique")
    if len(set(segment_ids)) != len(segment_ids):
        errors.append("Recognition queue segment IDs must be unique")
    if body_indices != [len(parts) - 1]:
        errors.append("Recognition queue must contain exactly one body_fallback, last")
    distinctive_scope_ranks = scope_ranks[:-1]
    if any(
        left < right
        for left, right in zip(
            distinctive_scope_ranks,
            distinctive_scope_ranks[1:],
            strict=False,
        )
        if left >= 0 and right >= 0
    ):
        errors.append(
            "Recognition queue distinctive estimated_scope must be large-to-small"
        )
    if live_gate.get("status") != "complete":
        errors.append("Live sequential gate must report status 'complete'")
    if live_gate.get("approved_count") != len(parts):
        errors.append("Live sequential gate must approve every queue part")
    if live_gate.get("approved_parts") != names:
        errors.append("Live sequential gate approved_parts must match queue order")

    raw_locks = lock_manifest.get("locks")
    if not isinstance(raw_locks, list):
        return [*errors, "part_lock_manifest.json must contain a locks array"]
    locks = [lock for lock in raw_locks if isinstance(lock, dict)]
    if len(locks) != len(raw_locks):
        errors.append("Every part lock must be a JSON object")
        return errors
    lock_names = [lock.get("part_name") for lock in locks]
    if lock_names != names:
        errors.append("Part lock order must exactly match the recognition queue")
    if len(locks) != len(parts):
        errors.append("Every recognized part, including body, must have one lock")
        return errors
    raw_reviews = review_manifest.get("reviews")
    if not isinstance(raw_reviews, list):
        errors.append("part_review_manifest.json must contain a reviews array")
        raw_reviews = []
    passed_reviews = {
        (
            review.get("part_name"),
            review.get("lock_fingerprint"),
            review.get("face_labels_sha256"),
        )
        for review in raw_reviews
        if isinstance(review, dict) and review.get("status") == "passed"
    }

    previous_labels: array[int] | None = None
    locked_ids: set[int] = set()
    expected_face_count: int | None = None
    final_lock_sha256: str | None = None
    for index, (part, lock, identity) in enumerate(
        zip(parts, locks, part_identities, strict=True)
    ):
        if identity is None:
            continue
        name, segment_id = identity
        lock_fingerprint = hashlib.sha256(
            json.dumps(lock, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if (
            name,
            lock_fingerprint,
            lock.get("face_labels_sha256"),
        ) not in passed_reviews:
            errors.append(
                f"Part {name} lacks a launcher-owned passing review for its exact lock"
            )
        errors.extend(
            validate_part_lock_selection_evidence(
                run_dir,
                part=part,
                lock=lock,
            )
        )
        if lock.get("segment_id") != segment_id:
            errors.append(f"Lock {index} segment_id does not match queue part {name}")
        if lock.get("order") != index:
            errors.append(f"Lock for {name} must have order {index}")
        unresolved = lock.get("unresolved_issues")
        if unresolved != []:
            errors.append(f"Part {name} cannot lock with unresolved issues")
        validation_artifacts = lock.get("validation_artifacts")
        if not isinstance(validation_artifacts, list) or not validation_artifacts:
            errors.append(f"Part {name} lock lacks validation artifacts")
        else:
            for artifact in validation_artifacts:
                artifact_path = _safe_run_artifact_path(run_dir, artifact)
                if artifact_path is None or not artifact_path.is_file():
                    errors.append(
                        f"Part {name} cites missing or unsafe validation artifact: "
                        f"{artifact!r}"
                    )

        labels_relative = lock.get("face_labels")
        labels_path = _safe_run_artifact_path(run_dir, labels_relative)
        if labels_path is None:
            errors.append(f"Part {name} lock lacks a safe face_labels path")
            continue
        labels, labels_error = _load_u32_labels(labels_path)
        if labels_error:
            errors.append(labels_error)
            continue
        assert labels is not None
        if expected_face_count is None:
            expected_face_count = len(labels)
        elif len(labels) != expected_face_count:
            errors.append(f"Part {name} lock changed the source face count")
        digest = _sha256_file(labels_path)
        if lock.get("face_labels_sha256") != digest:
            errors.append(f"Part {name} lock face-label SHA-256 does not match")
        final_lock_sha256 = digest

        allowed_ids = {other_segment_id, *locked_ids, segment_id}
        observed_ids = set(labels)
        future_ids = set(segment_ids[index + 1 :])
        if observed_ids & future_ids:
            errors.append(
                f"Part {name} lock assigns faces to unstarted segment IDs "
                f"{sorted(observed_ids & future_ids)}"
            )
        unexpected_ids = observed_ids - allowed_ids
        if unexpected_ids:
            errors.append(
                f"Part {name} lock contains unexpected segment IDs "
                f"{sorted(unexpected_ids)}"
            )
        accepted_face_count = sum(value == segment_id for value in labels)
        if accepted_face_count <= 0:
            errors.append(f"Part {name} lock is empty")
        if lock.get("accepted_face_count") != accepted_face_count:
            errors.append(f"Part {name} accepted_face_count does not match labels")
        if previous_labels is not None and len(labels) == len(previous_labels):
            for face_index, prior in enumerate(previous_labels):
                if prior in locked_ids and labels[face_index] != prior:
                    errors.append(
                        f"Part {name} changed a previously locked face at "
                        f"source face {face_index}"
                    )
                    break
        previous_labels = labels
        locked_ids.add(segment_id)

    if previous_labels is not None and other_segment_id in previous_labels:
        errors.append("Final body lock must consume every remaining other face")
    final_labels_path = run_dir / "final" / "face_labels.u32le"
    if final_labels_path.is_file() and final_lock_sha256 is not None:
        if _sha256_file(final_labels_path) != final_lock_sha256:
            errors.append("Final face labels must exactly match the last body lock")

    if queue.get("active_part") is not None:
        errors.append("Completed recognition queue must not retain an active part")
    for part in parts:
        if part.get("status") != "locked":
            name = part.get("name", part.get("part_name"))
            errors.append(f"Completed queue part {name!r} is not locked")
    return errors


def _write_json(path: Path, payload: object, *, mode: int = 0o600) -> None:
    """Atomically replace one JSON artifact with its final permissions."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(value).name
    suffix = Path(name).suffix
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    safe_suffix = suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix) else ""
    return f"{safe_stem or fallback}{safe_suffix}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "asset"
