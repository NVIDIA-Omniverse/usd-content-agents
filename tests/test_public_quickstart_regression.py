# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the public quickstart paths.

The public README quickstart for the Physics Agent previously failed
end-to-end with only NVIDIA_API_KEY in .env. Two distinct
foot-guns were responsible:

1. ``apps/physics_agent/configs/lightbulb.yaml`` -- the ``identify_asset``
   step had no ``renderer`` block, so it silently fell back to
   ``IDENTIFY_ASSET_DEFAULTS["renderer"]["backend"] = "remote"`` and tried
   to call NVCF, which the public user has not configured.

2. The agent-service ``docker-compose.yml`` files re-listed VLM provider
   API keys under ``environment:`` with ``${VAR:-}`` substitution. Compose
   substitution does NOT read ``env_file:`` -- it reads the project-dir
   ``.env`` (which defaults to the compose file's directory). With the
   user's ``.env`` at the repo root, every key resolved to an empty string
   and clobbered the values that ``env_file: path: ../../.env`` had just
   loaded into the container.

   The same substitution context applies to the ``${VAR:-default}`` lines
   for ``*_VLM_BACKEND``, ``*_VLM_MODEL``, ``*_LLM_BACKEND`` etc. -- those
   resolve to the *built-in default* (e.g. ``nim``) instead of the user's
   ``.env`` override. We don't strip those from ``environment:`` here
   because they're legitimate compose-level defaults; the public READMEs
   document ``--env-file .env`` so substitution finds the repo-root file.

These tests pin the fix in place: the public configs use the local OVRTX
backend by default, the compose files do not list provider API keys
under ``environment:`` (they flow through ``env_file:`` instead), and
the public READMEs document ``--env-file .env`` for the documented
``docker compose`` invocation.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
_PUBLIC_UTILS_PATH = Path(__file__).with_name("public_artifact_utils.py")
_PUBLIC_UTILS_SPEC = importlib.util.spec_from_file_location(
    "public_artifact_utils",
    _PUBLIC_UTILS_PATH,
)
assert _PUBLIC_UTILS_SPEC is not None
assert _PUBLIC_UTILS_SPEC.loader is not None
_public_utils = importlib.util.module_from_spec(_PUBLIC_UTILS_SPEC)
_PUBLIC_UTILS_SPEC.loader.exec_module(_public_utils)
public_doc_path = _public_utils.public_doc_path

SKILL_MIRROR_IGNORED_NAMES = {".DS_Store", "__pycache__"}
SKILL_MIRROR_IGNORED_SUFFIXES = {".pyc"}

# Public-shipping configs that a user with only NVIDIA_API_KEY should be
# able to run on a single GPU box. Each entry maps the config path to the
# pipeline steps whose `renderer` block must default to a local backend.
_PUBLIC_CONFIG_LOCAL_RENDER_STEPS: dict[str, tuple[str, ...]] = {
    "apps/physics_agent/configs/lightbulb.yaml": (
        "identify_asset",
        "build_dataset_usd",
    ),
}

_JOINT_PUBLIC_CONFIG = Path("apps/joint_agent/configs/byoa_joint_rigger.yaml")
_JOINT_PUBLIC_GUIDES = (
    Path("apps/joint_agent/README.md"),
    Path("apps/joint_agent/AGENTS.md"),
    Path("apps/joint_agent/CLAUDE.md"),
    Path(".agents/skills/joint-agent-cli/SKILL.md"),
)

# Compose files that load the repo-root .env via long-form `env_file`. The
# fix is to NOT list provider API keys under `environment:` for the same
# service, because Compose substitution can't see env_file contents.
_COMPOSE_FILES = (
    "apps/physics_agent_service/docker-compose.yml",
    "apps/material_agent_service/docker-compose.yml",
    "apps/joint_agent_service/docker-compose.yml",
    "apps/texture_agent_service/docker-compose.yml",
)

_FORBIDDEN_ENV_KEYS = (
    "NVIDIA_API_KEY",
    # Keep this split so the source-side scanner test does not match itself.
    "INFERENCE_" + "NVIDIA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    # NGC_API_KEY is also passed via env_file; keeping it under
    # `environment: ${NGC_API_KEY:-}` would clobber .env values too.
    "NGC_API_KEY",
)


def _service_uses_repo_root_env_file(service: dict) -> bool:
    """Return True if the service declares env_file pointing at ../../.env."""
    env_file = service.get("env_file")
    if env_file is None:
        return False
    if isinstance(env_file, str):
        return env_file.endswith("../../.env")
    if isinstance(env_file, list):
        for entry in env_file:
            if isinstance(entry, str) and entry.endswith("../../.env"):
                return True
            if isinstance(entry, dict) and str(entry.get("path", "")).endswith(
                "../../.env"
            ):
                return True
    return False


def _environment_keys(service: dict) -> set[str]:
    """Extract VAR names from the service's `environment:` block."""
    env = service.get("environment")
    if env is None:
        return set()
    keys: set[str] = set()
    if isinstance(env, list):
        for entry in env:
            if not isinstance(entry, str):
                continue
            name, _, _ = entry.partition("=")
            keys.add(name.strip())
    elif isinstance(env, dict):
        keys.update(str(k) for k in env.keys())
    return keys


def _skill_mirror_files(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and not set(path.relative_to(root).parts) & SKILL_MIRROR_IGNORED_NAMES
        and path.suffix.lower() not in SKILL_MIRROR_IGNORED_SUFFIXES
    )


@pytest.mark.parametrize(
    "config_relpath, steps",
    list(_PUBLIC_CONFIG_LOCAL_RENDER_STEPS.items()),
)
def test_public_config_render_steps_default_to_ovrtx(
    config_relpath: str, steps: tuple[str, ...]
) -> None:
    """Public configs must render with a local backend out of the box.

    A public user with only NVIDIA_API_KEY in .env (no NGC_API_KEY, no
    NVCF function id) must be able to run the quickstart end-to-end on a
    single GPU machine. Every step that renders has to use a local backend
    (ovrtx or warp) -- never `remote`, which requires NVCF.
    """
    config_path = REPO_ROOT / config_relpath
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for step_name in steps:
        step = config["steps"][step_name]
        renderer = step.get("renderer")
        assert renderer is not None, (
            f"{config_relpath}::{step_name} has no `renderer` block, so it "
            "will fall back to the global default which is `remote` "
            "(requires NVCF). Add `renderer: {backend: ovrtx, ...}`."
        )
        backend = renderer.get("backend")
        assert backend in {"ovrtx", "warp"}, (
            f"{config_relpath}::{step_name}.renderer.backend = {backend!r}; "
            "public quickstart configs must use a local backend so users "
            "with only NVIDIA_API_KEY can run end-to-end."
        )


def test_joint_public_config_surface_is_one_byoa_rigger_template() -> None:
    """Joint staging exposes one asset-neutral, candidate-driven config."""

    config_dir = REPO_ROOT / "apps/joint_agent/configs"
    if not config_dir.exists():
        pytest.skip("Joint Agent is not included in this staging release")

    public_configs = sorted(path.name for path in config_dir.glob("*.yaml"))
    assert public_configs == [_JOINT_PUBLIC_CONFIG.name]

    config_path = REPO_ROOT / _JOINT_PUBLIC_CONFIG
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)
    assert isinstance(config, dict)
    assert config["input"]["usd_path"] == "/absolute/path/to/your_asset.usd"
    assert ".data/" not in config_text
    assert "nvidia" + "_inference" not in config_text
    assert "llm" + "gateway" not in config_text

    steps = config["steps"]
    assert steps["identify_asset"]["renderer"]["backend"] == "remote"
    assert steps["build_dataset_usd"]["renderer"]["backend"] == "remote"
    assert steps["analyze_structure"]["llm"] == {
        "backend": "nim",
        "model": "qwen/qwen3.5-397b-a17b",
    }
    assert steps["predict"]["vlm"] == {
        "backend": "nim",
        "model": "qwen/qwen3.5-397b-a17b",
    }
    assert steps["predict"]["completion_retries"] == 3
    assert steps["build_dataset_prepare_dataset"]["prompt_profile"] == (
        "prop_articulation"
    )
    assert steps["infer_articulation_candidates"] == {
        "enabled": True,
        "output_key": "classification",
        "candidate_joint_types": ["revolute", "prismatic"],
        "adjudication": {
            "enabled": True,
            "reconcile_topology": True,
            "model_key": "vlm",
            "min_confidence": "high",
            "max_tokens": 8192,
            "max_images": 64,
            "require_source_images": True,
        },
        "vlm": {
            "backend": "nim",
            "model": "qwen/qwen3.5-397b-a17b",
        },
    }
    apply_step = steps["apply_joint_rigger"]
    assert apply_step["enabled"] is False
    assert apply_step["adapter"] == "owned_core"
    assert apply_step["apply_masses"] is False
    assert apply_step["apply_collision"] is False
    assert apply_step["articulation_candidates_path"].endswith(
        "articulation_candidates/articulation_candidates.json"
    )
    assert apply_step["output_usd_path"].endswith("joint_rigger/rigged.usdz")


def test_joint_service_public_defaults_are_remote_and_unbounded() -> None:
    """The public service should need no local renderer or fixed upload cap."""

    compose_path = REPO_ROOT / "apps/joint_agent_service/docker-compose.yml"
    if not compose_path.exists():
        pytest.skip("Joint Agent Service is not included in this staging release")

    compose_text = compose_path.read_text(encoding="utf-8")
    dockerfile_text = (REPO_ROOT / "apps/joint_agent_service/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "JA_RENDER_BACKEND=${JA_RENDER_BACKEND:-remote}" in compose_text
    assert "JA_MAX_UPLOAD_SIZE_MB=${JA_MAX_UPLOAD_SIZE_MB:-0}" in compose_text
    assert "runtime: nvidia" not in compose_text
    assert "JA_MAX_UPLOAD_SIZE_MB=0" in dockerfile_text
    assert "JA_RENDER_BACKEND=remote" in dockerfile_text
    assert "packages/usd_joint_rigger" not in dockerfile_text


def test_joint_public_guides_use_the_byoa_template() -> None:
    """Public Joint commands must not point users at internal asset configs."""

    if not (REPO_ROOT / "apps/joint_agent").exists():
        pytest.skip("Joint Agent is not included in this staging release")

    for relative_path in _JOINT_PUBLIC_GUIDES:
        path = public_doc_path(REPO_ROOT, relative_path)
        assert path.is_file(), relative_path
        text = path.read_text(encoding="utf-8")
        assert _JOINT_PUBLIC_CONFIG.as_posix() in text, relative_path
        assert "apps/joint_agent/configs/nova_carter.yaml" not in text
        assert "apps/joint_agent/configs/blender_ur10e.yaml" not in text


def test_joint_byoa_template_loads_after_input_and_authoring_opt_in(
    tmp_path: Path,
) -> None:
    """The public template must become executable after its documented edits."""

    if not (REPO_ROOT / "apps/joint_agent").exists():
        pytest.skip("Joint Agent is not included in this staging release")

    from joint_agent.config.unified_config import UnifiedPipelineConfigTask

    asset_path = tmp_path / "asset.usda"
    asset_path.write_text('#usda 1.0\ndef Xform "World" {}\n', encoding="utf-8")
    config = yaml.safe_load(
        (REPO_ROOT / _JOINT_PUBLIC_CONFIG).read_text(encoding="utf-8")
    )
    config["input"]["usd_path"] = str(asset_path)
    config["steps"]["apply_joint_rigger"]["enabled"] = True
    config_path = tmp_path / "joint_byoa.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    context = UnifiedPipelineConfigTask().run(
        {
            "config_path": str(config_path),
            "only_steps": [
                "infer_articulation_candidates",
                "apply_joint_rigger",
            ],
        }
    )

    assert context["steps_to_run"] == [
        "infer_articulation_candidates",
        "apply_joint_rigger",
    ]
    apply_config = context["step_configs"]["apply_joint_rigger"]
    assert apply_config["adapter"] == "owned_core"
    assert apply_config["articulation_candidates_path"] == str(
        tmp_path
        / ".joint-agent-byoa/articulation_candidates/articulation_candidates.json"
    )
    assert apply_config["output_usd_path"] == str(
        tmp_path / ".joint-agent-byoa/joint_rigger/rigged.usdz"
    )


@pytest.mark.parametrize("compose_relpath", _COMPOSE_FILES)
def test_compose_does_not_clobber_env_file_api_keys(compose_relpath: str) -> None:
    """env_file values must not be clobbered by `environment: ${VAR:-}` lines.

    Compose substitution does not read `env_file:` contents -- it reads
    the project-directory `.env` (which defaults to the compose file's
    parent directory). When the user's `.env` lives at the repo root,
    `${NVIDIA_API_KEY:-}` resolves to an empty string and overrides the
    value `env_file: ../../.env` just loaded into the container.

    The fix is to NOT list any provider API key under `environment:` for
    services that already pull from the repo-root env_file. The keys
    flow through env_file directly.
    """
    compose_path = REPO_ROOT / compose_relpath
    with compose_path.open(encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    offenders: list[str] = []
    for service_name, service in services.items():
        if not _service_uses_repo_root_env_file(service):
            continue
        env_keys = _environment_keys(service)
        for forbidden in _FORBIDDEN_ENV_KEYS:
            if forbidden in env_keys:
                offenders.append(f"{service_name}.{forbidden}")

    assert not offenders, (
        f"{compose_relpath} re-lists API keys under `environment:` for "
        "services that already use repo-root env_file. Compose substitution "
        "would clobber the env_file value with empty when the user's .env "
        "lives at the repo root. Drop these entries:\n  - " + "\n  - ".join(offenders)
    )


# Public READMEs that document a `docker compose -f apps/<svc>/docker-compose.yml`
# invocation. Each must use `--env-file .env` so Compose substitution resolves
# `${VAR:-default}` against the repo-root .env that the README told the user
# to populate -- without it, settings like `MA_VLM_BACKEND=openai` set in .env
# silently get clobbered by the compose default, even though the API key
# alongside them does flow through via the `env_file:` directive.
_PUBLIC_DOCKER_COMPOSE_READMES = (
    "README_PUBLIC.md",
    "apps/physics_agent_service/README.md",
    "apps/material_agent_service/README_PUBLIC.md",
    "apps/joint_agent_service/README.md",
    "apps/texture_agent_service/README.md",
)


@pytest.mark.parametrize("readme_relpath", _PUBLIC_DOCKER_COMPOSE_READMES)
def test_public_readme_compose_invocation_uses_env_file(readme_relpath: str) -> None:
    """Public docker-compose invocations must pass `--env-file .env`.

    Compose's variable substitution (``${VAR:-default}``) resolves against
    the project-directory ``.env``, which defaults to the compose file's
    parent directory. Without ``--env-file .env`` passed on the CLI, a
    user's ``PA_VLM_BACKEND=openai`` (or ``MA_VLM_MODEL=...``, etc.) in
    repo-root ``.env`` is silently ignored and Compose substitutes the
    built-in default -- the container ends up with the user's API key
    set but the wrong backend selected.

    This test scans every ``docker compose ... -f apps/<svc>/...``
    invocation in the public READMEs and requires that the same shell
    block / continued line includes ``--env-file .env``.
    """
    readme_path = public_doc_path(REPO_ROOT, readme_relpath)
    text = readme_path.read_text(encoding="utf-8")

    # Walk fenced ```bash blocks and look for `docker compose ... -f apps/`
    # invocations. Folded across line continuations (`\\\n`) so the test
    # tolerates the multi-line form the READMEs use.
    in_bash_block = False
    pending: list[str] = []
    invocations: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            # Toggle on/off any fenced block; treat shell-ish blocks as bash.
            fence = line.strip()
            if not in_bash_block and (fence == "```bash" or fence == "```sh"):
                in_bash_block = True
            elif in_bash_block:
                in_bash_block = False
                pending = []
            continue
        if not in_bash_block:
            continue
        # Strip leading prompt characters and inline comments.
        stripped = line.lstrip("# ").lstrip("$ ").rstrip()
        if stripped.endswith("\\"):
            pending.append(stripped[:-1].rstrip())
            continue
        pending.append(stripped)
        joined = " ".join(p for p in pending if p)
        pending = []
        if "docker compose" in joined and "-f apps/" in joined:
            invocations.append(joined)

    assert invocations, (
        f"{readme_relpath} declares no `docker compose -f apps/...` "
        "invocation. If the README intentionally moved away from compose, "
        "remove the entry from _PUBLIC_DOCKER_COMPOSE_READMES."
    )

    missing = [inv for inv in invocations if "--env-file" not in inv]
    assert not missing, (
        f"{readme_relpath} contains `docker compose` invocations that "
        "omit `--env-file .env`. Without that flag, Compose's `${{VAR:-...}}` "
        "substitution reads the compose-file-adjacent .env (which the user "
        "did not create) and silently falls back to built-in defaults, "
        "ignoring the user's repo-root .env overrides for backend/model "
        "variables. Offending lines:\n  - " + "\n  - ".join(missing)
    )


def test_material_large_scene_quickstart_uses_shipped_service_example() -> None:
    """Public large-scene docs must not point at root /examples, which do not ship."""
    readme_text = public_doc_path(REPO_ROOT, "README_PUBLIC.md").read_text(
        encoding="utf-8"
    )
    service_docs_text = (
        REPO_ROOT / "apps/material_agent_service/docs/api.md"
    ).read_text(encoding="utf-8")
    quickstart_path = (
        REPO_ROOT / "apps/material_agent_service/examples/large_scene/warehouse.usda"
    )

    assert quickstart_path.exists()
    assert "apps/material_agent_service/examples/large_scene/README.md" in readme_text
    assert "apps/material_agent_service/examples/large_scene/README.md" in (
        service_docs_text
    )
    assert "examples/material_agent_large_scene/README.md" not in readme_text
    assert "examples/material_agent_large_scene/README.md" not in service_docs_text


def test_texture_agent_cli_bootstraps_dotenv() -> None:
    """texture-agent must load repo-root .env before model calls need keys."""
    package_init = REPO_ROOT / "apps/texture_agent/texture_agent/__init__.py"
    cli_entrypoint = REPO_ROOT / "apps/texture_agent/texture_agent/cli.py"

    init_text = package_init.read_text(encoding="utf-8")
    cli_text = cli_entrypoint.read_text(encoding="utf-8")

    assert "from dotenv import load_dotenv" in init_text
    assert "load_dotenv()" in init_text
    assert "from dotenv import load_dotenv" in cli_text
    assert "load_dotenv()" in cli_text


def test_agent_skill_compatibility_mirrors() -> None:
    """Claude and Codex skill paths should link to the canonical tree."""
    canonical_skills = REPO_ROOT / ".agents/skills"
    claude_skills = REPO_ROOT / ".claude/skills"
    codex_skills = REPO_ROOT / ".codex/skills"

    assert canonical_skills.is_dir()
    assert claude_skills.is_dir()
    assert codex_skills.is_dir()
    assert claude_skills.is_symlink()
    assert codex_skills.is_symlink()
    assert claude_skills.resolve() == canonical_skills.resolve()
    assert codex_skills.resolve() == canonical_skills.resolve()

    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/sync_agent_skills.sh"), "--check"],
        cwd=REPO_ROOT,
        check=True,
    )

    canonical_files = _skill_mirror_files(canonical_skills)
    assert canonical_files

    for mirror in (claude_skills, codex_skills):
        mirror_files = _skill_mirror_files(mirror)
        assert mirror_files == canonical_files
        for rel_path in canonical_files:
            assert (mirror / rel_path).read_bytes() == (
                canonical_skills / rel_path
            ).read_bytes()

    assert (canonical_skills / "quickstart/SKILL.md").exists()


def test_sync_agent_skills_refuses_dirty_legacy_mirror(tmp_path: Path) -> None:
    """Legacy mirror-only files must be moved to .agents before replacement."""
    mirror = REPO_ROOT / ".claude/skills"
    original_target = Path(os.readlink(mirror))
    dirty_file = mirror / f"dirty-{tmp_path.name}.tmp"

    mirror.unlink()
    mirror.mkdir()
    dirty_file.write_text("mirror-only skill draft\n", encoding="utf-8")

    try:
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts/sync_agent_skills.sh")],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        shutil.rmtree(mirror, ignore_errors=True)
        mirror.symlink_to(original_target)

    assert result.returncode != 0
    assert "refusing to replace dirty skill mirror" in result.stderr
    subprocess.run(
        ["bash", str(REPO_ROOT / "scripts/sync_agent_skills.sh"), "--check"],
        cwd=REPO_ROOT,
        check=True,
    )


def test_deploy_collection_skill_metadata_exists() -> None:
    """The canonical skill tree should include Codex UI metadata."""
    canonical_skill = REPO_ROOT / ".agents/skills/deploy-collection/SKILL.md"
    metadata = REPO_ROOT / ".agents/skills/deploy-collection/agents/openai.yaml"

    assert canonical_skill.exists()
    assert metadata.exists()
    codex_metadata = REPO_ROOT / ".codex/skills/deploy-collection/agents/openai.yaml"
    assert codex_metadata.exists()
    assert codex_metadata.read_text(encoding="utf-8") == metadata.read_text(
        encoding="utf-8"
    )
