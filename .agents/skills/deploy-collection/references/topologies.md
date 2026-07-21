# Topologies

## Minimal Hosted Models

Use this as the default user path:

- CPU host: material, physics, texture services.
- GPU host: shared OVRTX rendering.
- Hosted model APIs: VLM, LLM, image generation, embeddings.
- The image-generation dependency can be shared by Texture Agent texture
  generation, Material Agent-generated reference images, and Material Agent
  generated material-library textures.

This requires the least GPU management while still validating shared rendering.

## External Endpoint Enterprise

Use this when the user already has model/render services:

- Set `provider: external`.
- Fill endpoint URLs in `deploy/collection/collection.yaml`.
- Put credentials in repo-root `.env`.
- Run `deploy.py plan`, `up`, then `smoke`.

## Local Workstation

Use this when dependencies are already running locally:

- OVRTX endpoint from containers should be `http://host.docker.internal:8001`.
- Keep `extra_hosts: host.docker.internal:host-gateway` in the agents Compose.
- Start dependencies separately before `deploy.py up`.

## Full Self-Hosted

Use this when the user wants every dependency hosted by the deployment strategy:

- OVRTX on a GPU host.
- VLM on A100/H100 or equivalent.
- Image generation on a GPU host for Texture Agent texture generation,
  Material Agent-generated reference images, and Material Agent-generated
  material-library textures.
- Optional LLM and embeddings on separate GPU hosts when resource pressure
  requires it.
- Agents remain CPU-only.

Full self-hosting can use Brev, local GPU machines, or any provider that exposes
stable HTTP endpoints.
