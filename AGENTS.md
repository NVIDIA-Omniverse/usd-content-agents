# Content Agents Agent Guide

This file provides public, repo-local guidance for coding agents working in
`NVIDIA-Omniverse/usd-content-agents`.

## Start Here

Use `README.md` for the canonical user quick start. Prefer the smallest
supported path for the user's goal:

1. Inspect `README.md`, `.env_example`, and the relevant app README under
   `apps/`.
2. Check prerequisites before installing: Python 3.12+, `uv`, Docker Compose
   v2.24+ for service mode, and an NVIDIA GPU/runtime when using the bundled
   rendering sidecars.
3. Keep API keys in `.env`. Do not print, commit, or paste secrets.
4. For local CLI mode from the repo root:

   ```bash
   uv venv --python=3.12
   source .venv/bin/activate
   uv pip install -e . -e apps/material_agent -e apps/physics_agent \
     -e apps/joint_agent -e apps/texture_agent -e apps/validation_agent
   ./scripts/fetch_build_resources.sh
   ```

5. Use a dry run before expensive VLM or rendering calls when the CLI supports
   it:

   ```bash
   material-agent run apps/material_agent/configs/unified_example.yaml --dry-run
   material-agent run apps/material_agent/configs/unified_example.yaml
   ```

## Repository Map

- `world_understanding/` - shared library code, tool registry, model wrappers,
  utility functions, and minimal agent framework.
- `apps/material_agent/` - Material Agent CLI and configs.
- `apps/material_agent_service/` - Material Agent REST service and client.
- `apps/physics_agent/` - Physics Agent CLI and configs.
- `apps/physics_agent_service/` - Physics Agent REST service and client.
- `apps/joint_agent/` - Joint Agent Research Preview CLI and configs.
- `apps/joint_agent_service/` - Joint Agent Research Preview REST service and client.
- `apps/texture_agent/` - Texture Agent CLI and configs.
- `apps/texture_agent_service/` - Texture Agent REST service and client.
- `apps/validation_agent/` - Validation Agent Research Preview CLI and configs.
- `apps/ovrtx_rendering_api/` - shared OVRTX rendering API sidecar.
- `agentic/` - Agentic Workflow preview workspace for Content Workbench and
  `content-workflow-cli`.
- `.agents/skills/` - canonical checked-in agent skills.
- `.claude/skills/` and `.codex/skills/` - compatibility mirrors of the
  canonical skill tree.

## Agent Skills

Start repo-level workflows from the repo root. For Agentic Workflow preview
sessions, start the agent from `agentic/` so it discovers the isolated preview
skills.

| Workflow | Codex skill | Claude skill | First command |
|---|---|---|---|
| Agentic Workflow preview | `agentic/.codex/skills/content-workflow-material` | `agentic/.claude/skills/content-workflow-material` | `content-workflow-cli materials assign --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml --output-dir runs/content-workflow-cli/ladder-codex` |
| Agentic batch launcher | `agentic/.codex/skills/content-workflow-cli` | `agentic/.claude/skills/content-workflow-cli` | `content-workflow-cli materials assign --usd path/to/asset.usd --materials-yaml path/to/materials.yaml` |
| Material CLI | `.codex/skills/material-agent-cli` | `.claude/skills/material-agent-cli` | `material-agent run apps/material_agent/configs/unified_example.yaml` |
| Physics CLI | `.codex/skills/physics-agent-cli` | `.claude/skills/physics-agent-cli` | `physics-agent run apps/physics_agent/configs/lightbulb.yaml` |
| Joint CLI (Research Preview) | `.codex/skills/joint-agent-cli` | `.claude/skills/joint-agent-cli` | `joint-agent run apps/joint_agent/configs/byoa_joint_rigger.yaml --dry-run` |
| Joint Gate 3 validation | `.codex/skills/joint-agent-validation` | `.claude/skills/joint-agent-validation` | Validate a published Joint Agent USD/USDZ with Gate 3A and Gate 3B. |
| Texture CLI | `.codex/skills/texture-agent-cli` | `.claude/skills/texture-agent-cli` | `texture-agent run apps/texture_agent/configs/texture_example.yaml` |
| Validation CLI (Research Preview) | `.codex/skills/validation-agent-cli` | `.claude/skills/validation-agent-cli` | `validation-agent run apps/validation_agent/examples/configs/steel_scaffold_behavior_refine_summary.yaml` |
| Material service | `.codex/skills/deploy-material-agent-docker` | `.claude/skills/deploy-material-agent-docker` | `docker compose --env-file .env -f apps/material_agent_service/docker-compose.yml up --build` |
| Physics service | `.codex/skills/deploy-physics-agent-docker` | `.claude/skills/deploy-physics-agent-docker` | `docker compose --env-file .env -f apps/physics_agent_service/docker-compose.yml up --build` |
| Joint service/client | `.codex/skills/joint-agent-client` | `.claude/skills/joint-agent-client` | `docker compose --env-file .env -f apps/joint_agent_service/docker-compose.yml up --build` |
| Texture service | `.codex/skills/deploy-texture-agent-docker` | `.claude/skills/deploy-texture-agent-docker` | `docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml up --build` |
| Full collection | `.codex/skills/deploy-collection` | `.claude/skills/deploy-collection` | `./deploy/collection/deploy.py plan && ./deploy/collection/deploy.py up` |
| USD utilities | `.codex/skills/flatten-usd`, `.codex/skills/print-usd`, `.codex/skills/render-usd` | `.claude/skills/flatten-usd`, `.claude/skills/print-usd`, `.claude/skills/render-usd` | Inspect, flatten, or render USD assets. |

## Public Backends

Public docs, configs, and examples should use only public model providers:

- `nim` with `NVIDIA_API_KEY`
- `openai` with `OPENAI_API_KEY`
- `anthropic` with `ANTHROPIC_API_KEY`
- `gemini` with `GOOGLE_API_KEY`

Do not hardcode credentials. Use placeholders in examples.

## Validation

Use lightweight checks before claiming a change is ready:

```bash
python3 -m pytest tests/test_public_quickstart_regression.py -q
python3 -m pytest tests/test_packaging_requests_declared.py tests/test_pyproject_fallback_version.py -q
```

If `pytest` is missing, install the development extras in the active
environment first:

```bash
uv pip install -e ".[dev]"
```

## Safety

- Do not commit `.env`, downloaded assets, generated textures, rendered images,
  session directories, service logs, or credentials.
- Use `--dry-run` before VLM or render calls when supported.
- Treat clean flags and commands that delete working directories as destructive.
- Keep changes scoped to the requested workflow and follow existing repo style.
