---
name: quickstart
description: Start one NVIDIA Content Agents REST service locally with Docker Compose. Use when the user asks for /quickstart, local POC setup, or single-service Material, Physics, Joint, or Texture startup.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - quickstart
  - docker
  - local-deploy
tools:
  - Shell
  - Docker
  - curl
  - Filesystem
compatibility: Requires Docker daemon, Docker Compose v2.24+, curl, and repo-root .env provider credentials.
---

# Content Agents Quickstart

## When to Use

- Use when the user asks for `/quickstart` or help starting a local
  Content Agents proof of concept.
- Use when the user wants exactly one Material, Physics, Joint, or Texture REST
  service from this repository's compose files.
- Use when the user needs a safe preflight, startup, health-check, and
  follow-up-command handoff for a local service.

## Scope

This skill starts one local Content Agents REST service from the repo root.
It is a thin wrapper over the existing per-agent Docker Compose files; do
not create a new compose stack for this workflow. Material Agent is the
recommended first POC.

- Start one target at a time: `material`, `physics`, `joint`, or `texture`.
- Use `deploy-collection` when the user wants to run Material, Physics, Joint,
  and Texture together or configure shared dependency endpoints.
- Use the service-specific deploy skills for local NIM sidecars, image-gen
  sidecars, embedding sidecars, multi-GPU layouts, or deliberate port changes.

## Limitations

- Keep secrets out of chat. Tell the user to edit `.env`; never ask them to
  paste API keys.
- Always use the repo-root `.env` file and keep `--env-file .env` in compose
  commands.
- Stop before startup when `.env` was just created or is missing. The user
  must edit provider credentials before any compose command starts services.
- Do not create a new compose stack for this workflow.
- Do not hold the Shell tool in long build or warm-up loops. Start detached,
  run only a short bounded health pass, and return clear follow-up commands
  when the service is still building or warming.

## Instructions

1. Infer the target from the user request, or ask for `material`, `physics`,
   `joint`, or `texture` when unclear.
2. Check prerequisites and create `.env` from `.env_example` only when the
   template exists.
3. Run the preflight commands and stop before startup when Docker, Compose,
   `.env`, required GPU/toolkit checks, or running-container conflicts are not
   resolved.
4. Start only after conflicts are resolved by stopping the owning stack or by
   restarting the selected target. If the user keeps an overlapping stack
   running, skip startup for the new target.
5. Start the selected compose stack with the matching prefix from the target
   table. Use a background handoff for first builds or any shell tool with a
   short timeout budget.
6. Run one short bounded health pass, including OVRTX for Material or Physics.
7. Return the service URL, docs URL, startup state, health result, and exact
   follow-up commands from the output format.

## Prerequisites

- Docker daemon reachable.
- Docker Compose v2.24+.
- Repo-root `.env` with the provider key for the selected backend. The default
  public backend usually needs `NVIDIA_API_KEY`.
- Joint also needs `RENDER_ENDPOINT`; its public Compose path uses remote
  rendering and does not require a local render GPU.
- Material and Physics require an RTX-capable NVIDIA render GPU plus NVIDIA
  Container Toolkit because their compose files start OVRTX.
- Recommended render-GPU capacity: Material is a 48 GB-class target and
  Physics is a 16 GB-class target. Smaller RTX GPUs, including 24 GB or 32 GB
  cards, may still be valid for small-scene POCs; warn the user and ask before
  proceeding rather than blocking solely on VRAM.
- A100, H100, H200, and V100 are model-serving GPUs, not render-GPU targets for
  this quickstart. Joint and Texture can run CPU-only with hosted dependencies.

If `.env` is missing, create it only when the template exists:

```bash
if [ -f .env ]; then
  echo "ENV_FILE=present"
elif [ -f .env_example ]; then
  cp .env_example .env
  echo "ENV_FILE=created; edit .env with one provider key before starting"
else
  echo "ENV_FILE=missing; create .env manually with provider keys"
fi
```

Treat `ENV_FILE=created` and `ENV_FILE=missing` as hard stops. Tell the user to
edit the repo-root `.env`, then rerun this quickstart after credentials are in
place.

## Preflight

Run this from the repo root and parse the `KEY=VALUE` lines:

```bash
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "DOCKER=ok"
else
  echo "DOCKER=missing_or_unreachable"
fi

if docker compose version >/dev/null 2>&1; then
  compose_version="$(docker compose version --short 2>/dev/null)"
  compose_version="${compose_version#v}"
  compose_major="${compose_version%%.*}"
  compose_rest="${compose_version#*.}"
  compose_minor="${compose_rest%%.*}"
  if ! [ "$compose_major" -eq "$compose_major" ] 2>/dev/null || ! [ "$compose_minor" -eq "$compose_minor" ] 2>/dev/null; then
    echo "COMPOSE=unparseable (${compose_version}; need 2.24+)"
  elif [ "$compose_major" -gt 2 ] 2>/dev/null || { [ "$compose_major" -eq 2 ] 2>/dev/null && [ "$compose_minor" -ge 24 ] 2>/dev/null; }; then
    echo "COMPOSE=ok (${compose_version})"
  else
    echo "COMPOSE=too_old (${compose_version}; need 2.24+)"
  fi
elif command -v docker-compose >/dev/null 2>&1; then
  echo "COMPOSE=legacy_v1"
else
  echo "COMPOSE=missing"
fi

if nvidia-smi >/dev/null 2>&1; then
  gpu_mem_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'BEGIN { max = 0 } $1 + 0 > max { max = $1 + 0 } END { if (max > 0) printf "%.0f", max }')"
  gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd '|' -)"
  echo "GPU=yes"
  echo "GPU_VRAM_MIB_MAX=${gpu_mem_mib:-unknown}"
  echo "GPU_NAMES=${gpu_names:-unknown}"
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "CONTAINER_TOOLKIT=ok"
  elif command -v nvidia-container-cli >/dev/null 2>&1 || command -v nvidia-container-runtime >/dev/null 2>&1; then
    echo "CONTAINER_TOOLKIT=installed_not_registered"
  else
    echo "CONTAINER_TOOLKIT=missing"
  fi
else
  echo "GPU=no"
  echo "GPU_VRAM_MIB_MAX=0"
  echo "GPU_NAMES=none"
  echo "CONTAINER_TOOLKIT=skipped"
fi

echo "RUNNING_CONTAINERS=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -E '^(material-agent-service|physics-agent-service|joint-agent-service|texture-agent-service|ovrtx-rendering-api|physics-ovrtx-rendering-api|ovrtx_rendering_api-ovrtx-rendering-api-1|vlm-nim|llm-nim|image-gen-nim)$' | paste -sd ',' -)"
```

Stop before startup when Docker or Compose is not ready. For Material or
Physics, also stop when there is no NVIDIA GPU, the NVIDIA Container Toolkit is
missing or not registered, or `GPU_NAMES` contains A100, H100, H200, or V100.
If VRAM is unknown or below the recommended capacity, warn the user and ask
whether to continue with a small-scene POC.

If `RUNNING_CONTAINERS` is non-empty, ask whether to stop the owning stack or
restart the selected target. Ports overlap across the per-agent compose files;
do not start a new target while overlapping containers continue to bind those
ports. If the user chooses to leave the existing stack running, skip startup and
report the existing-stack state.

## Target Table

| Target | Prefix | Base URL | Health | Docs |
|---|---|---|---|---|
| Material | `docker compose --env-file .env -f apps/material_agent_service/docker-compose.yml` | `http://localhost:8000` | `http://localhost:8000/health` | `http://localhost:8000/docs` |
| Physics | `docker compose --env-file .env -f apps/physics_agent_service/docker-compose.yml` | `http://localhost:8000` | `http://localhost:8000/health` | `http://localhost:8000/docs` |
| Joint (0.5 Research Preview) | `docker compose --env-file .env -f apps/joint_agent_service/docker-compose.yml` | `http://localhost:8000` | `http://localhost:8000/health` | `http://localhost:8000/docs` |
| Texture | `docker compose --env-file .env -f apps/texture_agent_service/docker-compose.yml` | `http://localhost:8001` | `http://localhost:8001/health` | `http://localhost:8001/docs` |

Infer the target from the user request. If unclear, ask:

```text
Which agent should I start: material (recommended), physics, joint, or texture?
```

## Start

Show a concise summary, then start the selected prefix.

Use the foreground form only when the images are already built or the Shell
tool timeout is known to be long enough for the first build:

```bash
<PREFIX> up -d --build
```

For first builds or short Shell tool timeouts, hand off the compose build in the
background and return immediately. Store transient logs under `logs/quickstart`
because that path is already ignored by git.

```bash
STARTUP_ROOT="logs/quickstart/<target>"
STARTUP_DIR="$STARTUP_ROOT/$(date +%Y%m%d-%H%M%S)"
STARTUP_LOG="$STARTUP_DIR/compose.log"
STARTUP_PID="$STARTUP_DIR/compose.pid"
STARTUP_LATEST="$STARTUP_ROOT/latest"
mkdir -p "$STARTUP_DIR" || { echo "STARTUP=blocked; cannot create $STARTUP_DIR"; exit 1; }
printf '%s\n' "$STARTUP_DIR" > "$STARTUP_LATEST" || { echo "STARTUP=blocked; cannot write $STARTUP_LATEST"; exit 1; }
nohup sh -c '<PREFIX> up -d --build' > "$STARTUP_LOG" 2>&1 &
pid="$!"
printf '%s\n' "$pid" > "$STARTUP_PID" || { echo "STARTUP=blocked; cannot write $STARTUP_PID"; exit 1; }
disown "$pid" 2>/dev/null || true
echo "STARTUP=background pid=$pid log=$STARTUP_LOG"
```

First build can take around 10 minutes. Material and Physics can take several
more minutes while OVRTX warms up. Do not wait through that whole period inside
one Shell tool call.

## Check

Run a short bounded check. If a background build is still running, report
`STARTUP=building` and return the log command instead of blocking.

```bash
<PREFIX> ps

STARTUP_ROOT="logs/quickstart/<target>"
STARTUP_LATEST="$STARTUP_ROOT/latest"
STARTUP_DIR="$(cat "$STARTUP_LATEST" 2>/dev/null || true)"
if [ -z "$STARTUP_DIR" ] || [ ! -d "$STARTUP_DIR" ]; then
  STARTUP_DIR="$(find "$STARTUP_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
fi
STARTUP_LOG="${STARTUP_DIR:+$STARTUP_DIR/compose.log}"
# STARTUP_PID is the PID-file path; pid below is the running process ID.
STARTUP_PID="${STARTUP_DIR:+$STARTUP_DIR/compose.pid}"

if [ -n "$STARTUP_PID" ] && [ -f "$STARTUP_PID" ]; then
  pid="$(cat "$STARTUP_PID" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "STARTUP=building pid=$pid log=$STARTUP_LOG"
    echo "LOG_COMMAND=tail -f $STARTUP_LOG"
    exit 0
  fi
fi

healthy=no
for i in 1 2 3 4 5 6; do
  if curl -fsS --max-time 5 <HEALTH_URL>; then
    healthy=yes
    break
  fi
  sleep 10
done

if [ "$healthy" = yes ]; then
  echo "SERVICE_HEALTH=healthy"
else
  echo "SERVICE_HEALTH=starting_or_unhealthy"
  if [ -s "$STARTUP_LOG" ]; then
    echo "STARTUP_LOG=$STARTUP_LOG"
    tail -n 80 "$STARTUP_LOG"
  fi
  <PREFIX> logs --no-color --tail=80 || true
fi
```

For Material or Physics, also run a bounded OVRTX check. If it is not healthy
yet, report it as warming unless logs show a clear error. OVRTX and Texture both
use port 8001; if Texture owns that port, report a port conflict instead of
counting `http://localhost:8001/health` as healthy for OVRTX.

```bash
ovrtx=warming
ovrtx_body=""
for i in 1 2 3 4 5 6; do
  ovrtx_body="$(curl -fsS --max-time 5 http://localhost:8001/health 2>/dev/null || true)"
  if printf '%s\n' "$ovrtx_body" | grep -Eq '"gpu_initialized"[[:space:]]*:[[:space:]]*true'; then
    ovrtx=healthy
    break
  fi
  sleep 10
done
echo "OVRTX_HEALTH=$ovrtx"
if [ "$ovrtx" != healthy ] && [ -n "$ovrtx_body" ]; then
  echo "OVRTX_HEALTH_BODY=$ovrtx_body"
fi
```

## Output Format

Report a concise summary with:

- Selected target and compose prefix.
- Service URL and docs URL.
- Startup state: `blocked`, `not_started`, `background_building`, `warming`,
  `healthy`, or `unhealthy`.
- Health result, including OVRTX health for Material or Physics.
- Any blocker found during preflight, including `.env` creation or unresolved
  port conflicts.
- Log file path or log command when startup is still building or warming.

Then return these follow-up commands:

```bash
<PREFIX> logs -f
<PREFIX> ps
<PREFIX> down
<PREFIX> down -v
```

When stopping a stack that was already running, use the compose file that owns
the running container. If ownership is ambiguous, inspect:

```bash
docker inspect --format '{{ index .Config.Labels "com.docker.compose.project.config_files" }}' <container-name>
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `DOCKER=missing_or_unreachable` | Docker is not installed, not running, or not reachable by the current user. | Stop and ask the user to start Docker or fix daemon access before retrying. |
| `COMPOSE=too_old` or `COMPOSE=legacy_v1` | Docker Compose is older than v2.24 or only legacy `docker-compose` is installed. | Ask the user to install Docker Compose v2.24+ and rerun preflight. |
| `ENV_FILE=created` or `ENV_FILE=missing` | The repo-root `.env` still needs provider credentials. | Stop startup, tell the user to edit `.env`, then rerun this skill. |
| Material or Physics reports no GPU/toolkit | The selected target needs the local render stack from Prerequisites. | Stop startup and explain that Joint or Texture can run CPU-only with hosted dependencies. |
| Joint cannot render | `RENDER_ENDPOINT` is missing or unavailable. | Set a reachable remote renderer endpoint in the repo-root `.env`, then retry. |
| Material or Physics reports low or unknown VRAM | Small-scene POCs may work, but the target is below recommended capacity. | Warn the user and ask before continuing; do not block solely on VRAM. |
| Service health stays unhealthy after the short check | The service is still building, warming, or a dependency is misconfigured. | Report the current state and show `<PREFIX> ps`, `curl -fsS <HEALTH_URL>`, and `<PREFIX> logs --no-color --tail=80`. |
| OVRTX is not immediately healthy | OVRTX can warm up after the service process starts. | Report `OVRTX_HEALTH=warming` and provide the logs command unless logs show a clear failure. |
| Texture is running while checking Material or Physics OVRTX | Texture and OVRTX both use port 8001. | Report the port owner conflict instead of counting `http://localhost:8001/health` as OVRTX healthy. |
| Ports conflict with another stack | The per-agent compose files share ports. | Stop or restart the owning stack before starting the new target; if the user leaves it running, skip startup. |
