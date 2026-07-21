---
name: deploy-collection
description: Deploy the full Content Agents package with a provider-neutral, endpoint-driven, multi-instance strategy. Use when the user asks to deploy all content agents, deploy material/physics/texture together, configure shared OVRTX rendering, choose VLM/LLM/image-gen/embedding endpoints, test the full package, or use Brev as an optional self-hosted dependency provider.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - collection
  - deployment
  - brev
tools:
  - Shell
  - Docker
  - Python
  - Filesystem
compatibility: Requires the repo Python environment, Docker Compose, endpoint URLs for shared dependencies, optional Brev CLI access for self-hosted dependency nodes, and provider credentials in the repo-root .env.
---

# Deploy Content Agents Collection

## When to Use

Use this skill to deploy the whole Content Agents package without Helm and
without a single monolithic Compose stack.

## Limitations

- Keep provider credentials in the repo-root `.env`; never print or commit
  secrets.
- Use Brev only when the user asks for self-hosted dependencies or a full
  validation topology.
- Use service-specific deploy skills when debugging one agent or dependency in
  isolation.

## Prerequisites

- Repo Python environment available for `deploy/collection/deploy.py`.
- Docker Compose available for local CPU-only service startup.
- Endpoint URLs or Brev deployment plans for OVRTX, VLM/LLM, image generation,
  and embedding dependencies selected by the user.

## Core Model

- Material, Physics, and Texture Agent services are CPU-only.
- OVRTX rendering is a shared GPU dependency required by Material and Physics.
- VLM, LLM, image generation, and embeddings are optional model dependencies.
- Material Agent Service uses the image-generation dependency for generated
  reference images and the opt-in generated-material-library workflow. Texture
  Agent texture generation, Material Agent-generated reference images, and
  Material Agent-generated material-library textures can share the same
  image-generation endpoint.
- Dependencies are configured by endpoint URL.
- Brev is optional. Use it only when the user asks for Brev or wants a full
  self-hosted validation topology.

## Instructions

1. Inspect `deploy/collection/collection.yaml`.
2. Ask for missing endpoint choices only if they cannot be inferred safely.
3. Keep secrets in the repo-root `.env`; never write API keys into YAML.
4. Run:

```bash
./deploy/collection/deploy.py plan
```

5. If the plan is valid, start the CPU-only agents:

```bash
./deploy/collection/deploy.py up
```

6. Check status or smoke:

```bash
./deploy/collection/deploy.py status
./deploy/collection/deploy.py smoke
```

7. Commit and push coherent checkpoints when making implementation changes.

## When To Read References

- For env vars and endpoint wiring, read `references/env-contract.md`.
- For choosing a topology, read `references/topologies.md`.
- For Brev-hosted dependencies, read `references/brev.md`.

## Existing Skills To Reuse

- Use `deploy-ovrtx-docker` when starting or debugging a standalone OVRTX
  renderer.
- Use `deploy-qwen-vlm-brev` when the user asks for a Brev-hosted Qwen VLM.
- Use `deploy-image-gen-brev` when the user asks for a Brev-hosted FLUX
  image-generation endpoint.
- Use `deploy-embeddings-brev` when the user asks for a Brev-hosted embedding
  endpoint for Material Agent prim clustering.
- Use `brev-cli` when creating, listing, accessing, port-forwarding, or cleaning
  up Brev instances.
- Use service-specific Docker skills only when debugging one agent in isolation.

Do not use Helm skills for this workflow unless the user explicitly changes the
deployment target to Kubernetes/Helm.

## Validation

Minimum validation for code changes:

```bash
source .venv/bin/activate
python -m py_compile deploy/collection/deploy.py
./deploy/collection/deploy.py plan
./deploy/collection/deploy.py render-env
docker compose --env-file deploy/collection/.collection.generated.env \
  -f deploy/collection/docker-compose.agents.yml config --quiet
```

Run Docker/Brev smoke tests only when the requested topology and credentials are
available.

## Output Format

Return the selected topology, generated endpoint configuration, validation
commands run, service URLs or pending dependencies, and any cleanup commands.

## Troubleshooting

- If planning fails, report the invalid field and the config file to edit.
- If Docker config or smoke checks fail, include the failing command, relevant
  log tail, and the next retry or cleanup command.
