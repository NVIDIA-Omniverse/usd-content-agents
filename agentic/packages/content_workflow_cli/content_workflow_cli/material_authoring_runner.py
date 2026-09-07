# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin launcher for the skill-routed material-authoring workflow."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from content_agent_workflows.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from content_agent_workflows.common.usd_cli_session import (
    WorkflowUsdCliSession,
    validated_ovrtx_render_metadata,
)
from PIL import Image, UnidentifiedImageError

from .runner import (
    CLAUDE_EXECUTION_CLI,
    CLAUDE_EXECUTION_SDK,
    CODEX_SANDBOX_WORKSPACE_WRITE,
    RUNNER_CLAUDE,
    RUNNER_CODEX,
    ParentUsdCliCapability,
    find_repo_root,
    parent_usd_cli_prompt_contract,
    run_child_agent,
    start_parent_usd_cli_capability,
    stop_parent_usd_cli_capability,
)
from .trace import TraceWriter, build_trace

MATERIAL_AUTHORING_REQUEST_SCHEMA_VERSION = (
    "content-workflow-material-authoring-request.v1"
)
MATERIAL_AUTHORING_RESULT_SCHEMA_VERSION = (
    "content-workflow-material-authoring-result.v1"
)
MATERIAL_AUTHORING_VISUAL_ASSESSMENT_SCHEMA_VERSION = (
    "content-workflow-material-authoring-visual-assessment.v1"
)
MATERIAL_AUTHORING_RENDER_PROVENANCE_SCHEMA_VERSION = (
    "content-workflow-material-authoring-render-provenance.v1"
)
MATERIAL_AUTHORING_WORKFLOW_SKILL = "content-workflow-material-authoring"
MATERIAL_AUTHORING_MODES = ("create", "refine", "re_author")
MATERIAL_AUTHORING_PROFILES = (
    "auto",
    "preview_surface",
    "openpbr_materialx",
    "omnipbr_mdl",
)
MATERIAL_AUTHORING_REQUIRED_SKILLS = (
    MATERIAL_AUTHORING_WORKFLOW_SKILL,
    "material-generation",
    "image-generation",
    "usd-cli",
)


@dataclass(frozen=True)
class MaterialAuthoringRunConfig:
    """One fresh wrapper-launched material-authoring run."""

    repo_root: Path
    mode: str
    intent: str
    output_dir: Path | None
    source_material_usd: Path | None = None
    source_material_prim_path: str | None = None
    source_asset: Path | None = None
    reference_images: tuple[Path, ...] = ()
    material_profile: str = "auto"
    allow_approximation: bool = False
    max_attempts: int = 3
    prompt_file: Path | None = None
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
    child_timeout_seconds: float = 1800.0
    dry_run: bool = False
    json_output: bool = False
    agent_cwd: Path | None = None


@dataclass(frozen=True)
class _MaterialAuthoringChildConfig:
    """Adapter for the shared child-agent launcher."""

    child_launch_profile: ClassVar[str] = "materials.author"
    repo_root: Path
    usd_path: Path
    reference_images: list[Path]
    reference_files: list[Path] | None
    runner: str
    model: str | None
    model_reasoning_effort: str | None
    codex_base_url: str | None
    codex_sandbox_mode: str
    codex_config: dict[str, object] | None
    claude_config: dict[str, object] | None
    claude_permission_mode: str
    claude_max_turns: int | None
    claude_execution_mode: str
    child_timeout_seconds: float
    agent_cwd: Path | None
    usd_cli_server_url: str | None = None
    parent_usd_cli_session_identity: Path | None = None
    parent_usd_cli_session_identity_sha256: str | None = None
    workflow_skill: str = MATERIAL_AUTHORING_WORKFLOW_SKILL


def add_material_authoring_subcommand(subparsers: Any) -> None:
    """Register ``materials author`` and its ``generate`` alias."""

    author = subparsers.add_parser(
        "author",
        aliases=["generate"],
        help="Create, refine, or re-author one material through workflow skills.",
    )
    author.add_argument("--mode", required=True, choices=MATERIAL_AUTHORING_MODES)
    author.add_argument(
        "--usd",
        type=Path,
        help="Source material USD for refine or re_author mode.",
    )
    author.add_argument(
        "--material-path",
        help="Absolute material prim path in --usd.",
    )
    author.add_argument(
        "--source-asset",
        type=Path,
        help="Optional USD asset used as visual/semantic context.",
    )
    intent = author.add_mutually_exclusive_group(required=True)
    intent.add_argument("--prompt", help="Material-authoring intent as text.")
    intent.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 file containing the material-authoring intent.",
    )
    author.add_argument(
        "--reference-image",
        action="append",
        type=Path,
        default=[],
        help="Optional appearance reference image; repeat for multiple images.",
    )
    author.add_argument(
        "--material-profile",
        choices=MATERIAL_AUTHORING_PROFILES,
        default="auto",
    )
    author.add_argument("--allow-approximation", action="store_true")
    author.add_argument("--max-attempts", type=int, default=3)
    author.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh run directory; defaults under .local-runs/content-workflow-cli.",
    )
    author.add_argument(
        "--repo-root",
        type=Path,
        help="Repository root. Defaults to the current Git worktree.",
    )
    author.add_argument(
        "--runner",
        choices=[RUNNER_CODEX, RUNNER_CLAUDE],
        default=RUNNER_CODEX,
    )
    author.add_argument("--model")
    author.add_argument("--model-reasoning-effort")
    author.add_argument(
        "--codex-base-url",
        default=os.getenv("CONTENT_AGENT_CODEX_BASE_URL"),
    )
    author.add_argument(
        "--codex-sandbox-mode",
        choices=[CODEX_SANDBOX_WORKSPACE_WRITE],
        default=CODEX_SANDBOX_WORKSPACE_WRITE,
    )
    author.add_argument(
        "--claude-permission-mode",
        choices=["default", "acceptEdits", "bypassPermissions", "plan"],
        default="default",
    )
    author.add_argument("--claude-max-turns", type=int)
    author.add_argument(
        "--claude-execution-mode",
        choices=[CLAUDE_EXECUTION_SDK, CLAUDE_EXECUTION_CLI],
        default=CLAUDE_EXECUTION_SDK,
    )
    author.add_argument("--child-timeout", type=float, default=1800.0)
    author.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze the request and prompt without launching a child agent.",
    )
    author.add_argument("--json", dest="json_output", action="store_true")
    author.set_defaults(handler=_handle_material_authoring)


def _handle_material_authoring(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root()
    prompt_file = (
        _resolve_file(args.prompt_file, label="prompt file")
        if args.prompt_file is not None
        else None
    )
    intent = (
        prompt_file.read_text(encoding="utf-8")
        if prompt_file is not None
        else args.prompt
    )
    return run_material_authoring(
        MaterialAuthoringRunConfig(
            repo_root=repo_root,
            mode=args.mode,
            intent=intent,
            output_dir=args.output_dir,
            source_material_usd=args.usd,
            source_material_prim_path=args.material_path,
            source_asset=args.source_asset,
            reference_images=tuple(args.reference_image),
            material_profile=args.material_profile,
            allow_approximation=args.allow_approximation,
            max_attempts=args.max_attempts,
            prompt_file=prompt_file,
            runner=args.runner,
            model=args.model,
            model_reasoning_effort=args.model_reasoning_effort,
            codex_base_url=args.codex_base_url,
            codex_sandbox_mode=args.codex_sandbox_mode,
            claude_permission_mode=args.claude_permission_mode,
            claude_max_turns=args.claude_max_turns,
            claude_execution_mode=args.claude_execution_mode,
            child_timeout_seconds=args.child_timeout,
            dry_run=args.dry_run,
            json_output=args.json_output,
        )
    )


def _resolve_file(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_source_material(
    usd_value: Path | None,
    prim_value: str | None,
) -> dict[str, str] | None:
    if usd_value is None and prim_value is None:
        return None
    if usd_value is None or prim_value is None:
        raise ValueError("--usd and --material-path must be supplied together")
    usd_path = _resolve_file(
        usd_value,
        label="source material USD",
    )
    if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        raise ValueError("--usd must name a USD file")
    prim_path = prim_value.strip()
    if not prim_path.startswith("/") or prim_path == "/" or prim_path.endswith("/"):
        raise ValueError("--material-path must be an absolute prim path")
    return {
        "usd_path": str(usd_path),
        "usd_sha256": file_sha256(usd_path),
        "material_prim_path": prim_path,
    }


def _resolve_reference_images(value: tuple[Path, ...]) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for index, raw_path in enumerate(value):
        path = _resolve_file(
            raw_path,
            label=f"reference image {index + 1}",
        )
        if path.suffix.lower() not in {
            ".bmp",
            ".gif",
            ".jpeg",
            ".jpg",
            ".png",
            ".tif",
            ".tiff",
            ".webp",
        }:
            raise ValueError(f"reference image has unsupported extension: {path}")
        images.append({"path": str(path), "sha256": file_sha256(path)})
    return images


def _build_request(config: MaterialAuthoringRunConfig) -> dict[str, Any]:
    mode = config.mode.strip()
    if mode not in MATERIAL_AUTHORING_MODES:
        raise ValueError(f"mode must be one of {MATERIAL_AUTHORING_MODES}")
    intent = config.intent.strip()
    if not intent:
        raise ValueError("intent must be a non-empty string")
    source_material = _resolve_source_material(
        config.source_material_usd,
        config.source_material_prim_path,
    )
    if mode == "create" and source_material is not None:
        raise ValueError("mode='create' must not include --usd or --material-path")
    if mode in {"refine", "re_author"} and source_material is None:
        raise ValueError(f"mode={mode!r} requires --usd and --material-path")

    source_asset = (
        _resolve_file(config.source_asset, label="source asset")
        if config.source_asset is not None
        else None
    )
    if source_asset is not None and source_asset.suffix.lower() not in {
        ".usd",
        ".usda",
        ".usdc",
        ".usdz",
    }:
        raise ValueError("source_asset must be a USD file")

    preview_input = source_asset
    preview_input_kind = "source_asset"
    if preview_input is None:
        preview_input = (
            config.repo_root
            / "apps"
            / "material_agent"
            / "data"
            / "templates"
            / "thumbnail_template.usd"
        ).resolve()
        preview_input_kind = "canonical_fixture"
        if not preview_input.is_file():
            raise FileNotFoundError(
                f"canonical material preview fixture is missing: {preview_input}"
            )

    material_profile = config.material_profile.strip()
    if material_profile not in MATERIAL_AUTHORING_PROFILES:
        raise ValueError(
            f"material_profile must be one of {MATERIAL_AUTHORING_PROFILES}"
        )
    if isinstance(config.max_attempts, bool):
        raise ValueError("max_attempts must be an integer")
    max_attempts = config.max_attempts
    if not 1 <= max_attempts <= 8:
        raise ValueError("max_attempts must be between 1 and 8")
    if not isinstance(config.allow_approximation, bool):
        raise ValueError("allow_approximation must be true or false")
    prompt_file = (
        _resolve_file(config.prompt_file, label="prompt file")
        if config.prompt_file is not None
        else None
    )
    prompt_source = (
        {
            "path": str(prompt_file),
            "sha256": file_sha256(prompt_file),
        }
        if prompt_file is not None
        else None
    )
    return {
        "schema_version": MATERIAL_AUTHORING_REQUEST_SCHEMA_VERSION,
        "workflow": "materials.author",
        "mode": mode,
        "intent": intent,
        "prompt_source": prompt_source,
        "source_material": source_material,
        "source_asset": (
            {"path": str(source_asset), "sha256": file_sha256(source_asset)}
            if source_asset is not None
            else None
        ),
        "preview_input": {
            "kind": preview_input_kind,
            "usd_path": str(preview_input),
            "usd_sha256": file_sha256(preview_input),
        },
        "reference_images": _resolve_reference_images(config.reference_images),
        "material_profile": material_profile,
        "allow_approximation": config.allow_approximation,
        "max_attempts": max_attempts,
        "required_skills": list(MATERIAL_AUTHORING_REQUIRED_SKILLS),
    }


def _default_output_dir(repo_root: Path, mode: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        repo_root
        / ".local-runs"
        / "content-workflow-cli"
        / f"material-authoring-{mode}-{timestamp}-{os.getpid()}"
    )


def _fresh_output_dir(config: MaterialAuthoringRunConfig) -> Path:
    candidate = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else _default_output_dir(config.repo_root, config.mode)
    )
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"output directory must be fresh: {candidate}")
    candidate.mkdir(parents=True)
    return candidate


def _required_skill_paths(repo_root: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in MATERIAL_AUTHORING_REQUIRED_SKILLS:
        candidates = (
            repo_root / "agentic" / ".agents" / "skills" / name / "SKILL.md",
            repo_root / ".agents" / "skills" / name / "SKILL.md",
        )
        skill_path = next(
            (path.resolve() for path in candidates if path.is_file()), None
        )
        if skill_path is None:
            raise FileNotFoundError(
                f"required material-authoring skill is missing: {name}"
            )
        paths[name] = str(skill_path)
    return paths


def material_authoring_skills_available(repo_root: Path | None = None) -> bool:
    """Return whether this distribution contains the complete internal workflow.

    Parser construction must stay side-effect free, so source-tree discovery is
    based on this module's location rather than invoking Git through
    ``find_repo_root``. Installed wheels naturally resolve to a root without
    internal skills and therefore keep the command hidden.
    """

    candidate_root = (
        repo_root.resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )

    try:
        _required_skill_paths(candidate_root)
    except FileNotFoundError:
        return False
    return True


def _build_prompt(
    *,
    request_path: Path,
    request_sha256: str,
    request: dict[str, Any],
    skill_paths: dict[str, str],
    usd_cli_server_url: str | None = None,
) -> str:
    task = {
        "schema_version": "content-agents.skill-routed-task.v1",
        "workflow": "materials.author",
        "request_path": str(request_path),
        "request_sha256": request_sha256,
        "required_skills": list(MATERIAL_AUTHORING_REQUIRED_SKILLS),
        "required_skill_paths": skill_paths,
        "mode": request["mode"],
        "usd_cli_server_url": usd_cli_server_url,
    }
    return f"""You are the single long-running child for an agentic material-authoring workflow.

Load and follow every skill in this immutable task, beginning with
`content-workflow-material-authoring`:
```json
{json.dumps(task, indent=2)}
```

Read the complete frozen request from `request_path`; do not modify it or any
source/reference input. The workflow skill owns attempts, visual review,
acceptance, and terminal artifacts. Atomic skills own only their named
capabilities. Do not invoke a fixed Material Agent workflow or the legacy
rendered-refinement controller.

The launcher owns the usd-cli daemon lifecycle. When `usd_cli_server_url` is
non-null, pass `--server <usd_cli_server_url>` before the verb in every raw
`usd-cli` or `usd-cli-tel` command. Never start, stop, or replace that daemon.

Write all attempts and evidence beneath the run directory. Before exiting,
write `final_summary.json` exactly as required by the workflow skill. Your final
response must be concise JSON containing only `status` and
`final_summary_path`.
"""


def _child_config(
    config: MaterialAuthoringRunConfig,
    request: dict[str, Any],
    request_path: Path,
    *,
    usd_cli_server_url: str | None = None,
) -> _MaterialAuthoringChildConfig:
    source_material = request.get("source_material")
    source_asset = request.get("source_asset")
    usd_path = (
        Path(source_material["usd_path"])
        if isinstance(source_material, dict)
        else Path(source_asset["path"])
        if isinstance(source_asset, dict)
        else request_path
    )
    return _MaterialAuthoringChildConfig(
        repo_root=config.repo_root,
        usd_path=usd_path,
        reference_images=[Path(item["path"]) for item in request["reference_images"]],
        reference_files=None,
        runner=config.runner,
        model=config.model,
        model_reasoning_effort=config.model_reasoning_effort,
        codex_base_url=config.codex_base_url,
        codex_sandbox_mode=config.codex_sandbox_mode,
        codex_config=config.codex_config,
        claude_config=config.claude_config,
        claude_permission_mode=config.claude_permission_mode,
        claude_max_turns=config.claude_max_turns,
        claude_execution_mode=config.claude_execution_mode,
        child_timeout_seconds=config.child_timeout_seconds,
        agent_cwd=config.agent_cwd,
        usd_cli_server_url=usd_cli_server_url,
    )


def _contained_file(run_dir: Path, raw_path: object, *, label: str) -> Path:
    candidate = Path(str(raw_path)).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (run_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(run_dir.resolve()) or not resolved.is_file():
        raise ValueError(f"{label} must be a file inside the run directory")
    return resolved


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return data


def _source_reference_matches(
    candidate: object,
    expected: object,
) -> bool:
    if not isinstance(candidate, dict) or not isinstance(expected, dict):
        return False
    try:
        candidate_path = Path(str(candidate["usd_path"])).expanduser().resolve()
        expected_path = Path(str(expected["usd_path"])).expanduser().resolve()
    except (KeyError, OSError):
        return False
    return (
        candidate_path == expected_path
        and candidate.get("usd_sha256") == expected.get("usd_sha256")
        and candidate.get("material_prim_path") == expected.get("material_prim_path")
    )


def _validate_material_package(
    run_dir: Path,
    package: dict[str, Any],
    *,
    accepted_attempt: int,
    request: dict[str, Any],
) -> tuple[Path, str]:
    try:
        from material_agent.material_library_generation import (
            MaterialPackageAuthoringError,
            load_material_package,
        )
    except ImportError as exc:
        raise RuntimeError(
            "material package validation requires content-workflow-cli[materials]"
        ) from exc

    attempt_dir = (run_dir / "attempts" / f"attempt-{accepted_attempt:03d}").resolve()
    paths = {
        key: _contained_file(run_dir, package.get(key), label=key)
        for key in (
            "material_usd_path",
            "materials_manifest_path",
            "authoring_manifest_path",
        )
    }
    for key, path in paths.items():
        if not path.is_relative_to(attempt_dir):
            raise ValueError(f"{key} must belong to the accepted attempt")
    material_usd_path = paths["material_usd_path"]
    materials_manifest_path = paths["materials_manifest_path"]
    authoring_manifest_path = paths["authoring_manifest_path"]
    try:
        validated = load_material_package(authoring_manifest_path.parent)
    except MaterialPackageAuthoringError as exc:
        raise ValueError(f"accepted material package is invalid: {exc}") from exc
    if validated.material_usd_path != material_usd_path:
        raise ValueError("authoring manifest material_usd_path is inconsistent")
    if validated.materials_manifest_path != materials_manifest_path:
        raise ValueError("authoring manifest materials_manifest_path is inconsistent")
    if validated.authoring_manifest_path != authoring_manifest_path:
        raise ValueError("authoring manifest path is inconsistent")
    authoring_manifest = _load_json_mapping(
        authoring_manifest_path,
        label="authoring_manifest_path",
    )
    authoring_request = authoring_manifest.get("request")
    if not isinstance(authoring_request, dict):
        raise ValueError("authoring manifest request must be a mapping")
    mode = request.get("mode")
    expected_operation = "modify" if mode == "refine" else "create"
    if (
        authoring_request.get("operation") != expected_operation
        or validated.operation.value != expected_operation
    ):
        raise ValueError("accepted package operation does not match the frozen request")
    source_material = request.get("source_material")
    if mode == "refine":
        if (
            not _source_reference_matches(
                authoring_request.get("source"), source_material
            )
            or authoring_request.get("provenance_source") is not None
        ):
            raise ValueError(
                "accepted refine package does not bind the frozen source material"
            )
    elif mode == "re_author":
        if authoring_request.get("source") is not None or not _source_reference_matches(
            authoring_request.get("provenance_source"), source_material
        ):
            raise ValueError(
                "accepted re-author package does not bind the frozen source material"
            )
    elif mode == "create":
        if (
            authoring_request.get("source") is not None
            or authoring_request.get("provenance_source") is not None
        ):
            raise ValueError("accepted create package must not bind a source material")
    else:
        raise ValueError("frozen request has an invalid material-authoring mode")
    requested_profile = request.get("material_profile")
    authored_profile = authoring_request.get("material_profile")
    if requested_profile == "auto":
        if (
            authored_profile not in set(MATERIAL_AUTHORING_PROFILES) - {"auto"}
            or validated.material_profile != authored_profile
        ):
            raise ValueError(
                "accepted package does not record a valid resolved material profile"
            )
    elif (
        authored_profile != requested_profile
        or validated.material_profile != requested_profile
    ):
        raise ValueError("accepted package resolved to a different material profile")
    if authoring_request.get("target_prim_paths") not in ([], ()):
        raise ValueError("accepted standalone material package must not assign targets")
    expected_semantics = (
        "generation_hints" if mode == "create" else "literal_shader_values"
    )
    if authoring_request.get("recipe_semantics") != expected_semantics:
        raise ValueError(
            "accepted package recipe semantics do not match the frozen request"
        )
    return material_usd_path, validated.material_prim_path


def _attest_reviewed_render(
    run_dir: Path,
    *,
    usd_cli_session: WorkflowUsdCliSession,
    input_path: Path,
    material_usd_path: Path,
    material_prim_path: str,
    render_path: Path,
    settings: dict[str, Any],
    expected_metadata: dict[str, Any],
    target_prim_path: str,
    camera_path: str,
) -> Path:
    """Re-render one reviewed image through the parent-owned trusted session."""

    width = int(settings["width"])
    height = int(settings["height"])
    raw_dir = usd_cli_session.prepare_raw_directory()
    attestation_dir = Path(
        tempfile.mkdtemp(prefix="material-render-attestation-", dir=raw_dir)
    )
    token = secrets.token_hex(12)
    parent_render_path = attestation_dir / f"{render_path.stem}-{token}.png"
    response_path = attestation_dir / f"{render_path.stem}-{token}-response.json"
    attestation_path = attestation_dir / f"{render_path.stem}-{token}.json"

    input_sha256 = file_sha256(input_path)
    usd_cli_session.open(input_path, read_only=False, force_reload=True)
    material_name = material_prim_path.rsplit("/", 1)[-1]
    usd_cli_session.run_json(
        [
            "material",
            target_prim_path,
            "--library",
            str(material_usd_path),
            "--library-prim",
            material_prim_path,
            "--name",
            material_name,
        ]
    )
    usd_cli_session.run_json(["camera", "use", camera_path])
    response = usd_cli_session.run_json(
        [
            "render",
            "--photoreal",
            "--mode",
            "quality",
            "--res",
            f"{width}x{height}",
            "--output",
            str(parent_render_path),
        ]
    )
    try:
        parent_metadata = validated_ovrtx_render_metadata(response)
    except RuntimeError as exc:
        raise ValueError(
            "parent-attested render lacks complete OVRTX provenance"
        ) from exc
    for key in (
        "backend",
        "ovrtx_render_mode",
        "ovrtx_num_sensor_updates",
        "active_aov",
        "renderer_identity",
    ):
        if parent_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                "reviewed render metadata does not match the parent-attested render"
            )
    if not parent_render_path.is_file():
        raise ValueError("parent-owned usd-cli render did not produce an image")
    if file_sha256(input_path) != input_sha256:
        raise ValueError("parent render modified the frozen preview input")
    try:
        with (
            Image.open(render_path) as reviewed,
            Image.open(parent_render_path) as attested,
        ):
            reviewed_rgba = reviewed.convert("RGBA")
            attested_rgba = attested.convert("RGBA")
            if (
                reviewed_rgba.size != (width, height)
                or attested_rgba.size != (width, height)
                or reviewed_rgba.tobytes() != attested_rgba.tobytes()
            ):
                raise ValueError(
                    "reviewed image pixels do not match the parent-attested render"
                )
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("parent-attested render is not a valid image") from exc

    atomic_write_json(response_path, response, within=run_dir)
    atomic_write_json(
        attestation_path,
        {
            "schema_version": (
                "content-workflow-material-authoring-parent-render-attestation.v1"
            ),
            "input_usd_path": str(input_path),
            "input_usd_sha256": input_sha256,
            "material_usd_path": str(material_usd_path),
            "material_usd_sha256": file_sha256(material_usd_path),
            "material_prim_path": material_prim_path,
            "target_prim_path": target_prim_path,
            "camera_path": camera_path,
            "reviewed_render_path": str(render_path),
            "reviewed_render_sha256": file_sha256(render_path),
            "parent_render_path": str(parent_render_path),
            "parent_render_sha256": file_sha256(parent_render_path),
            "parent_response_path": str(response_path),
            "parent_response_sha256": file_sha256(response_path),
            "render_settings": dict(settings),
        },
        within=run_dir,
    )
    return attestation_path


def _validate_render_provenance(
    run_dir: Path,
    attempt_dir: Path,
    render: dict[str, Any],
    *,
    render_path: Path,
    material_usd_path: Path,
    material_prim_path: str,
    request: dict[str, Any],
    usd_cli_session: WorkflowUsdCliSession,
) -> None:
    provenance_path = _contained_file(
        run_dir,
        render.get("provenance_path"),
        label="visual assessment render provenance",
    )
    if not provenance_path.is_relative_to(attempt_dir):
        raise ValueError("render provenance must belong to the accepted attempt")
    provenance = _load_json_mapping(
        provenance_path,
        label="visual assessment render provenance",
    )
    if (
        provenance.get("schema_version")
        != MATERIAL_AUTHORING_RENDER_PROVENANCE_SCHEMA_VERSION
    ):
        raise ValueError("render provenance has the wrong schema_version")
    preview_input = request.get("preview_input")
    if not isinstance(preview_input, dict):
        raise ValueError("frozen request is missing preview_input")
    input_path = (
        Path(
            _nonempty_string(
                provenance.get("input_usd_path"), label="render provenance input USD"
            )
        )
        .expanduser()
        .resolve()
    )
    expected_input_path = (
        Path(str(preview_input.get("usd_path"))).expanduser().resolve()
    )
    if (
        input_path != expected_input_path
        or provenance.get("input_usd_sha256") != preview_input.get("usd_sha256")
        or file_sha256(input_path) != preview_input.get("usd_sha256")
    ):
        raise ValueError("render provenance does not bind the frozen preview input")
    if Path(
        str(provenance.get("material_usd_path"))
    ).expanduser().resolve() != material_usd_path or provenance.get(
        "material_usd_sha256"
    ) != file_sha256(material_usd_path):
        raise ValueError("render provenance does not bind the accepted material USD")
    if Path(
        str(provenance.get("render_path"))
    ).expanduser().resolve() != render_path or provenance.get(
        "render_sha256"
    ) != file_sha256(render_path):
        raise ValueError("render provenance does not bind the reviewed image")
    response_path = _contained_file(
        run_dir,
        provenance.get("response_path"),
        label="render provenance response",
    )
    if not response_path.is_relative_to(attempt_dir):
        raise ValueError("render response must belong to the accepted attempt")
    if provenance.get("response_sha256") != file_sha256(response_path):
        raise ValueError("render response digest does not match provenance")
    response = _load_json_mapping(response_path, label="render provenance response")
    try:
        metadata = validated_ovrtx_render_metadata(response)
    except RuntimeError as exc:
        raise ValueError("render response lacks complete OVRTX provenance") from exc
    settings = provenance.get("render_settings")
    if not isinstance(settings, dict):
        raise ValueError("render provenance requires render_settings")
    for key in (
        "backend",
        "ovrtx_render_mode",
        "ovrtx_num_sensor_updates",
        "active_aov",
    ):
        if settings.get(key) != metadata[key]:
            raise ValueError("render settings do not match executed OVRTX metadata")
    width = settings.get("width")
    height = settings.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width < 1
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height < 1
    ):
        raise ValueError("render provenance requires positive image dimensions")
    with Image.open(render_path) as image:
        if image.width != width or image.height != height:
            raise ValueError("render dimensions do not match provenance")
    target_prim_path = _nonempty_string(
        provenance.get("target_prim_path"),
        label="render provenance target prim",
    )
    if not target_prim_path.startswith("/") or target_prim_path == "/":
        raise ValueError("render provenance target prim must be an absolute prim path")
    camera_path = _nonempty_string(
        provenance.get("camera_path"),
        label="render provenance camera path",
    )
    if not camera_path.startswith("/") or camera_path == "/":
        raise ValueError("render provenance camera path must be an absolute prim path")
    if preview_input.get("kind") == "canonical_fixture" and (
        target_prim_path != "/Root/Sphere" or camera_path != "/Root/thumbnail_CAM"
    ):
        raise ValueError(
            "canonical material preview must use the frozen sphere and camera"
        )
    _attest_reviewed_render(
        run_dir,
        usd_cli_session=usd_cli_session,
        input_path=input_path,
        material_usd_path=material_usd_path,
        material_prim_path=material_prim_path,
        render_path=render_path,
        settings=settings,
        expected_metadata=metadata,
        target_prim_path=target_prim_path,
        camera_path=camera_path,
    )


def _validate_visual_assessment(
    run_dir: Path,
    raw_path: object,
    *,
    accepted_attempt: int,
    request_sha256: str,
    material_usd_path: Path,
    material_prim_path: str,
    request: dict[str, Any],
    usd_cli_session: WorkflowUsdCliSession,
) -> None:
    assessment_path = _contained_file(
        run_dir,
        raw_path,
        label="visual_assessment_path",
    )
    attempt_dir = (run_dir / "attempts" / f"attempt-{accepted_attempt:03d}").resolve()
    if not assessment_path.is_relative_to(attempt_dir):
        raise ValueError("visual_assessment_path must belong to the accepted attempt")
    assessment = _load_json_mapping(
        assessment_path,
        label="visual_assessment_path",
    )
    if (
        assessment.get("schema_version")
        != MATERIAL_AUTHORING_VISUAL_ASSESSMENT_SCHEMA_VERSION
    ):
        raise ValueError("visual assessment has the wrong schema_version")
    if assessment.get("request_sha256") != request_sha256:
        raise ValueError("visual assessment does not bind the frozen request")
    if assessment.get("attempt") != accepted_attempt:
        raise ValueError("visual assessment does not bind the accepted attempt")
    if assessment.get("verdict") != "accepted":
        raise ValueError("accepted material requires an accepted visual verdict")
    if assessment.get("material_usd_sha256") != file_sha256(material_usd_path):
        raise ValueError("visual assessment does not bind the accepted material USD")
    _nonempty_string(assessment.get("summary"), label="visual assessment summary")
    renders = assessment.get("renders")
    if not isinstance(renders, list) or not renders:
        raise ValueError("visual assessment requires at least one reviewed render")
    for index, render in enumerate(renders):
        if not isinstance(render, dict):
            raise ValueError(f"visual assessment render {index} must be a mapping")
        render_path = _contained_file(
            run_dir,
            render.get("path"),
            label=f"visual assessment render {index}",
        )
        if not render_path.is_relative_to(attempt_dir):
            raise ValueError("reviewed renders must belong to the accepted attempt")
        if render.get("sha256") != file_sha256(render_path):
            raise ValueError("visual assessment render digest does not match")
        try:
            with Image.open(render_path) as image:
                image.verify()
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("visual assessment render is empty")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("visual assessment render is not a valid image") from exc
        _validate_render_provenance(
            run_dir,
            attempt_dir,
            render,
            render_path=render_path,
            material_usd_path=material_usd_path,
            material_prim_path=material_prim_path,
            request=request,
            usd_cli_session=usd_cli_session,
        )


def _validate_final_summary(
    run_dir: Path,
    *,
    request: dict[str, Any],
    request_sha256: str,
    usd_cli_session: WorkflowUsdCliSession | None = None,
) -> tuple[dict[str, Any], int]:
    summary_path = run_dir / "final_summary.json"
    data = _load_json_mapping(summary_path, label="final_summary.json")
    if data.get("schema_version") != MATERIAL_AUTHORING_RESULT_SCHEMA_VERSION:
        raise ValueError("final_summary.json has the wrong schema_version")
    if data.get("request_sha256") != request_sha256:
        raise ValueError("final_summary.json does not bind the frozen request")
    status = data.get("status")
    if status not in {"accepted", "rejected", "blocked"}:
        raise ValueError("final_summary.json has an invalid status")
    _nonempty_string(data.get("summary"), label="final summary")
    if status == "accepted":
        accepted_attempt = data.get("accepted_attempt")
        if (
            not isinstance(accepted_attempt, int)
            or isinstance(accepted_attempt, bool)
            or accepted_attempt < 1
            or accepted_attempt > request.get("max_attempts", 0)
        ):
            raise ValueError("accepted final summary requires accepted_attempt")
        package = data.get("material_package")
        if not isinstance(package, dict):
            raise ValueError("accepted final summary requires material_package")
        material_usd_path, material_prim_path = _validate_material_package(
            run_dir,
            package,
            accepted_attempt=accepted_attempt,
            request=request,
        )
        if usd_cli_session is None:
            raise ValueError(
                "accepted material requires parent-owned render attestation"
            )
        _validate_visual_assessment(
            run_dir,
            data.get("visual_assessment_path"),
            accepted_attempt=accepted_attempt,
            request_sha256=request_sha256,
            material_usd_path=material_usd_path,
            material_prim_path=material_prim_path,
            request=request,
            usd_cli_session=usd_cli_session,
        )
        return data, 0
    if (
        data.get("accepted_attempt") is not None
        or data.get("material_package") is not None
    ):
        raise ValueError(
            "rejected or blocked final summary must null accepted_attempt and "
            "material_package"
        )
    visual_assessment_path = data.get("visual_assessment_path")
    if visual_assessment_path is not None:
        _contained_file(
            run_dir,
            visual_assessment_path,
            label="visual_assessment_path",
        )
    return data, 1 if status == "rejected" else 3


def _emit_terminal_failure(
    *,
    run_dir: Path,
    payload: dict[str, Any],
    trace_writer: TraceWriter,
    status: str,
    json_output: bool,
    error: BaseException | None = None,
) -> None:
    payload["status"] = status
    atomic_write_json(run_dir / "workflow_result.json", payload)
    trace_writer.write(
        "workflow_failed",
        phase="material_authoring_skill_routed",
        summary="Material-authoring workflow stopped before a validated handoff.",
        artifacts=[str(run_dir / "request.json"), str(run_dir / "agent_prompt.md")],
        data={"status": status, "error_type": type(error).__name__ if error else None},
    )
    build_trace(run_dir)
    _print_result(payload, json_output=json_output)


def _inputs_unchanged(
    request: dict[str, Any],
    *,
    request_path: Path,
    request_sha256: str,
) -> bool:
    if file_sha256(request_path) != request_sha256:
        return False
    prompt_source = request.get("prompt_source")
    if (
        isinstance(prompt_source, dict)
        and file_sha256(prompt_source["path"]) != prompt_source["sha256"]
    ):
        return False
    source = request.get("source_material")
    if (
        isinstance(source, dict)
        and file_sha256(source["usd_path"]) != source["usd_sha256"]
    ):
        return False
    source_asset = request.get("source_asset")
    if (
        isinstance(source_asset, dict)
        and file_sha256(source_asset["path"]) != source_asset["sha256"]
    ):
        return False
    preview_input = request.get("preview_input")
    if (
        not isinstance(preview_input, dict)
        or file_sha256(preview_input["usd_path"]) != preview_input["usd_sha256"]
    ):
        return False
    return all(
        file_sha256(reference["path"]) == reference["sha256"]
        for reference in request["reference_images"]
    )


def _print_result(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Run directory: {payload['run_dir']}")
    print(f"Request: {payload['request_path']}")
    print(f"Prompt: {payload['prompt_path']}")
    print(f"Status: {payload['status']}")


def _preview_usd(config: MaterialAuthoringRunConfig, request: dict[str, Any]) -> Path:
    del config
    preview_input = request.get("preview_input")
    if not isinstance(preview_input, dict):
        raise ValueError("frozen request is missing preview_input")
    return Path(_nonempty_string(preview_input.get("usd_path"), label="preview USD"))


def _usd_cli_server_url(run_dir: Path) -> str:
    state_path = run_dir / ".usd-cli" / "server.json"
    state = _load_json_mapping(state_path, label="parent-owned usd-cli daemon state")
    host = state.get("host")
    port = state.get("port")
    if host != "127.0.0.1" or not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeError("parent-owned usd-cli daemon has invalid loopback state")
    return f"http://{host}:{port}"


def run_material_authoring(config: MaterialAuthoringRunConfig) -> int:
    """Freeze one request, launch the workflow child, and validate its handoff."""

    request = _build_request(config)
    run_dir = _fresh_output_dir(config)
    request["run_dir"] = str(run_dir)
    request_path = run_dir / "request.json"
    atomic_write_json(request_path, request)
    request_sha256 = file_sha256(request_path)
    skill_paths = _required_skill_paths(config.repo_root)
    prompt = _build_prompt(
        request_path=request_path,
        request_sha256=request_sha256,
        request=request,
        skill_paths=skill_paths,
    )
    prompt_path = run_dir / "agent_prompt.md"
    atomic_write_text(prompt_path, prompt)
    trace_writer = TraceWriter(run_dir)
    trace_writer.write(
        "workflow_started",
        phase="material_authoring_skill_routed",
        summary="Prepared one agentic material-authoring child task.",
        artifacts=[str(request_path), str(prompt_path)],
        data={"mode": request["mode"], "dry_run": config.dry_run},
    )

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "request_path": str(request_path),
        "request_sha256": request_sha256,
        "prompt_path": str(prompt_path),
        "status": "prepared" if config.dry_run else "running",
        "dry_run": config.dry_run,
    }
    if config.dry_run:
        trace_writer.write(
            "workflow_dry_run",
            phase="material_authoring_skill_routed",
            summary="Dry run stopped before launching the child agent.",
        )
        build_trace(run_dir)
        _print_result(payload, json_output=config.json_output)
        return 0

    preview_usd = _preview_usd(config, request)
    source_material = request.get("source_material")
    usd_cli_inputs = [preview_usd]
    if isinstance(source_material, dict):
        usd_cli_inputs.append(Path(source_material["usd_path"]))
    child_config = _child_config(config, request, request_path)
    parent_usd_cli_capability: ParentUsdCliCapability | None = None
    run_error: BaseException | None = None
    try:
        parent_usd_cli_capability = start_parent_usd_cli_capability(
            config=child_config,
            run_dir=run_dir,
            workflow="materials.author",
            session_workflow="material-authoring",
            input_roots=tuple(usd_cli_inputs),
            initial_scene=preview_usd,
            timeout_seconds=max(config.child_timeout_seconds, 900.0),
        )
        usd_cli_session = parent_usd_cli_capability.session
        usd_cli_server_url = parent_usd_cli_capability.server_url
        child_config = replace(
            child_config,
            usd_cli_server_url=usd_cli_server_url,
            parent_usd_cli_session_identity=parent_usd_cli_capability.identity_path,
            parent_usd_cli_session_identity_sha256=(
                parent_usd_cli_capability.identity_sha256
            ),
        )
        child_output_path = run_dir / "raw" / "material_authoring_child_output.jsonl"
        child_final_path = run_dir / "raw" / "material_authoring_child_final.json"
        prompt = _build_prompt(
            request_path=request_path,
            request_sha256=request_sha256,
            request=request,
            skill_paths=skill_paths,
            usd_cli_server_url=usd_cli_server_url,
        )
        prompt += parent_usd_cli_prompt_contract(parent_usd_cli_capability)
        atomic_write_text(prompt_path, prompt)
        child_output_path.parent.mkdir(parents=True, exist_ok=True)
        trace_writer.write(
            "usd_cli_ready",
            phase="material_authoring_skill_routed",
            summary="Started the parent-owned usd-cli daemon with OVRTX ready.",
            artifacts=[
                str(preview_usd),
                *(
                    [str(parent_usd_cli_capability.readiness.artifact_path)]
                    if parent_usd_cli_capability.readiness.artifact_path is not None
                    else []
                ),
            ],
            data={
                "usd_cli_session_id": usd_cli_session.session_id,
                "usd_cli_server_url": usd_cli_server_url,
            },
        )
        returncode = run_child_agent(
            config=child_config,
            prompt=prompt,
            run_dir=run_dir,
            child_output_path=child_output_path,
            child_final_path=child_final_path,
            prompt_image_inputs=[
                {"label": f"material reference {index}", "path": item["path"]}
                for index, item in enumerate(request["reference_images"], start=1)
            ],
            bridge_artifact_prefix="material_authoring_skill_routed",
        )
        trace_writer.write(
            "child_finished",
            phase="material_authoring_skill_routed",
            summary="Material-authoring child exited.",
            artifacts=[str(child_output_path), str(child_final_path)],
            data={"returncode": returncode},
        )
        if returncode != 0:
            _emit_terminal_failure(
                run_dir=run_dir,
                payload=payload,
                trace_writer=trace_writer,
                status="child_failed",
                json_output=config.json_output,
            )
            return returncode
        try:
            if not _inputs_unchanged(
                request,
                request_path=request_path,
                request_sha256=request_sha256,
            ):
                raise RuntimeError(
                    "material-authoring child modified an immutable input"
                )
            summary, returncode = _validate_final_summary(
                run_dir,
                request=request,
                request_sha256=request_sha256,
                usd_cli_session=usd_cli_session,
            )
        except Exception as exc:
            _emit_terminal_failure(
                run_dir=run_dir,
                payload=payload,
                trace_writer=trace_writer,
                status="validation_failed",
                json_output=config.json_output,
                error=exc,
            )
            raise
        parent_attestations = sorted(
            path
            for directory in (run_dir / "raw").glob("material-render-attestation-*")
            for path in directory.glob("*.json")
        )
        payload.update(
            {
                "status": summary["status"],
                "final_summary_path": str(run_dir / "final_summary.json"),
                "parent_render_attestations": [
                    str(path)
                    for path in parent_attestations
                    if not path.name.endswith("-response.json")
                ],
            }
        )
        atomic_write_json(run_dir / "workflow_result.json", payload)
        trace_writer.write(
            "workflow_finished",
            phase="material_authoring_skill_routed",
            summary="Validated the agentic material-authoring handoff.",
            artifacts=[
                str(run_dir / "final_summary.json"),
                *payload["parent_render_attestations"],
            ],
            data={"status": summary["status"]},
        )
        build_trace(run_dir)
        _print_result(payload, json_output=config.json_output)
        return returncode
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        if parent_usd_cli_capability is not None:
            try:
                stop_parent_usd_cli_capability(parent_usd_cli_capability)
            except BaseException as cleanup_exc:
                if run_error is None:
                    raise
                cleanup_summary = str(cleanup_exc) or type(cleanup_exc).__name__
                run_error.add_note(
                    f"usd-cli session cleanup also failed: {cleanup_summary}"
                )


__all__ = [
    "MATERIAL_AUTHORING_REQUEST_SCHEMA_VERSION",
    "MATERIAL_AUTHORING_RENDER_PROVENANCE_SCHEMA_VERSION",
    "MATERIAL_AUTHORING_RESULT_SCHEMA_VERSION",
    "MATERIAL_AUTHORING_VISUAL_ASSESSMENT_SCHEMA_VERSION",
    "MATERIAL_AUTHORING_WORKFLOW_SKILL",
    "MaterialAuthoringRunConfig",
    "add_material_authoring_subcommand",
    "material_authoring_skills_available",
    "run_material_authoring",
]
