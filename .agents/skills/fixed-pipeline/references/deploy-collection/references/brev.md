# Brev Option

Brev is an optional provider for dependency endpoints. It should not be required
for the default deployment path.

## When To Use

- The user explicitly asks for Brev.
- The user wants to validate a full self-hosted deployment.
- The user has no available local GPU for OVRTX or model sidecars.

## Role Mapping

- `render`: use standalone OVRTX on an L40S, RTX PRO Server 6000, or similar.
- `vlm`: use A100/H100 for Qwen/NIM/vLLM OpenAI-compatible VLM. The validated
  low-cost path is `Qwen/Qwen2.5-VL-7B-Instruct` with
  `vllm/vllm-openai:v0.8.5.post1` on Denvr A100.
- `llm`: use a lighter text model host when separate from VLM.
- `image_gen`: use FLUX image generation host for Texture Agent. The validated
  Brev path is `nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant`
  on `g6e.xlarge`, forwarded to local Docker-reachable port `8016`.
- `embeddings`: use material clustering embedding model host.

## Operator Steps

1. Use `brev ls` and `brev create --dry-run` before creating instances.
2. Start the role service on the Brev instance.
3. Port-forward or expose the service endpoint.
4. Paste the reachable URL into `deploy/collection/collection.yaml`.
5. Run `./deploy/collection/deploy.py plan`.
6. Run `./deploy/collection/deploy.py smoke` after agents start.

Use `$fixed-pipeline` and the existing `brev-cli`, `deploy-ovrtx-docker`,
`deploy-qwen-vlm-brev`, `deploy-image-gen-brev`, and
`deploy-embeddings-brev` references for the low-level commands.

## Docker Host Networking

`brev port-forward` binds the forwarded port to localhost. Docker containers
cannot reach that through `host.docker.internal`. For local agent containers,
open a standard SSH forward bound to all host interfaces:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 0.0.0.0:8013:localhost:8001 content-render
```

Binding to `0.0.0.0` exposes the forwarded service on every host interface.
Use it only when local Docker containers must reach the Brev endpoint through
`host.docker.internal`; otherwise keep `brev port-forward` or bind SSH to
`127.0.0.1`. If `0.0.0.0` is required, restrict access with host firewall rules.

Then set the render endpoint to:

```yaml
endpoint: http://host.docker.internal:8013
```
