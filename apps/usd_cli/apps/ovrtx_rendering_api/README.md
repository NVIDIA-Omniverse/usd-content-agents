# OVRTX rendering adapter

This is a managed infrastructure component used by Content Agents. It is not a
standalone product or independently operated public service.

The managed adapter exposes an HTTP transport over the in-tree `usd-cli` component's
local OVRTX backend (`usd_core.render.ovrtx.OvRTXRenderBackend`). It pairs with the
`remote` renderer in `usd_core`; Content Agents infrastructure can direct a usd-cli
client to the adapter when rendering must run on a managed RTX host.

It speaks the contract `usd_core/render/remote.py` expects. All endpoints except `/live`
require `Authorization: Bearer <OVRTX_API_KEY>`:

- `POST /render` — request `{usdz_base64 | usd, cameras, image_width, image_height, mode}`
  (`mode` defaults to `quality`/RT2) → response
  `{results: [{camera, image_base64}]}` (base64 PNG per camera).
- `POST /render/upload` — multipart transport: raw (optionally gzipped) binary USDZ in
  `file`, JSON render parameters in the `params` form field.
- `POST /render/negotiate` / `PUT /blobs/{sha256}` / `POST /render/manifest` — the
  content-addressed transport (upload-dedup plan Phase 2, advertised as the `cas`
  `/live` feature): the client sends the bundle's per-file sha256 manifest, uploads
  only the blobs this node is missing (each optionally gzip/zstd-compressed, digest
  verified on write), and renders by manifest; the service hardlinks a staging tree
  from its LRU blob store. A blob evicted mid-flight answers 409 with the missing
  list — a partially-materialized scene is never rendered.
- `POST /physics/simulate` — remote execution of a caller-authored physics scenario:
  multipart raw (optionally gzipped) flattened USD/USDA/USDZ in `file`,
  `{body_pattern, duration_s, dt, sample_fps,
  compression}` in `params` → `{trajectory: [[t, pose7, vel6], …], n_bodies, n_steps}`.
  Runs the real ovphysx solver in its own isolated venv; this is how
  `usd-cli physics simulate` executes on hosts without a local ovphysx platform.
- `GET /health` — authenticated initialization detail.
- `GET /ready` — 200 only after GPU warm-up; 503 while initializing or failed
  (`/physics/simulate` is independent of render warm-up).
- `GET /live` — public, non-sensitive process liveness. Also reports
  `protocol_version` (see below).

`status` moves through `initializing`, `ready`, or `failed`; a bounded failure summary is
retained. A restart retries failed initialization.

## Version handshake

The service bakes in `usd_core.remote_protocol.PROTOCOL_VERSION` from the Content Agents checkout it
was built from and reports it on `GET /live` and in `/health`. Every usd-cli client
verifies it before uploading anything and **refuses to run** against a backend whose
version differs (or that predates version reporting), with an error asking for a
backend re-deploy. After pulling client changes that bump `PROTOCOL_VERSION`, rebuild
and re-deploy this service from the same checkout. `usd-cli remote configure <url>` shows
the match status. To disable the check at your own risk, set
`remote_verify_version = false` directly under the config file's `[render]` section,
or export `USD_CLI_RENDER_REMOTE_VERIFY_VERSION=false`. This bypasses only the
handshake; it cannot make an older request/response schema compatible.

## Security and deployment boundary

This service parses untrusted native USD data and consumes expensive GPU resources. It is
not an internet-facing application by itself. Deploy it behind an approved TLS-terminating
ingress with firewall policy, rate limits, request logging/metrics, and secret injection.
Never send proprietary scenes over plaintext HTTP. The application enforces a 70 MiB body
limit, exactly one scene source, at most 8 cameras, at most 4096×4096/16.7M pixels, a
single bounded GPU queue, and an 1800-second request timeout. Tighten these defaults for
the deployment. Follow the Content Agents root `SECURITY.md` and deployment policy.

## Requirements

Linux + NVIDIA RTX GPU + driver/Vulkan. The service process needs `usd-core` (pxr) to parse
and author the stage; the actual RTX render runs in an isolated, exactly qualified
runtime (`ovrtx==0.4.1.364340`, `ovstage==0.1.1.355824`, and `warp-lang==1.16.0`
from `https://pypi.nvidia.com`). Docker uses the build-time pre-provisioned
`/opt/ovrtx_venv`, baked from hash-pinned wheel URLs, and sets
`WU_OVRTX_AUTO_PROVISION=0`, so the image never installs or replaces it at runtime.
In supported unprovisioned bare-metal environments, first-use provisioning occurs only
when `render.ovrtx_auto_install = true` or `WU_OVRTX_AUTO_PROVISION=1`; otherwise the
backend fails fast with pre-provisioning instructions. Physics
simulation likewise runs in an isolated `ovphysx` venv (`ovphysx==0.4.13`, pre-provisioned
in the Docker image) — ovphysx bundles its own OpenUSD and cannot share a process with
`usd-core`; the solver itself runs on CPU.
The adapter package pins its matching `usd-cli` component version; update and validate
the pair together through the Content Agents repository gate.

## Run with Docker (recommended)

```bash
# from apps/usd_cli in a Content Agents checkout
export OVRTX_API_KEY="$(openssl rand -hex 32)"
export CUDA_BASE_IMAGE="nvidia/cuda:12.6.3-runtime-ubuntu24.04@sha256:<approved-digest>"
docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up --build
```

Needs BuildKit: the image selects its OVRTX wheel from `TARGETARCH`, which only
BuildKit populates, and the build fails closed rather than silently baking the
x86_64 wheel on arm64. Docker Engine 23+ and Compose v2 use BuildKit by default;
with `DOCKER_BUILDKIT=0` or Compose v1, set `DOCKER_BUILDKIT=1`.

Needs the NVIDIA container runtime. OVRTX is pinned and pre-baked at image build; runtime
package installation is disabled. The image runs as an unprivileged user and records its
Python package inventory at `/app/python-inventory.json`; the component workflow generates
the image SBOM and fails on high or critical known vulnerabilities. The repository variable
`USD_CLI_CUDA_BASE_IMAGE` must contain the security-approved immutable
`...ubuntu24.04@sha256:<64-hex-digest>` base used by that gate. The Dockerfile also rejects
any supplied base whose `python3` is outside the packages' supported 3.11/3.12 range.

## Run locally

```bash
# Run from apps/usd_cli in a Content Agents checkout. Keep usd-exchange as the
# environment's only native pxr provider.
uv pip install -e ".[cli,server]" -e apps/ovrtx_rendering_api \
  --overrides requirements/usd-exchange-override.txt
OVRTX_API_KEY=<strong-secret> uvicorn service.main:app --host 127.0.0.1 --port 8000
```

## Point a usd-cli client at it

Easiest — let the CLI write the config and verify connectivity in one step:

```bash
usd-cli remote serve-cmd            # print the exact commands to run on the remote GPU box
usd-cli remote serve-cmd --bare     # bare-metal (uvicorn) variant instead of Docker
usd-cli remote configure https://gpu-host     # reads the key from OVRTX_API_KEY
```

Keep the bearer key in the environment. `usd-cli remote configure` accepts keys only
from `OVRTX_API_KEY` (or `USD_CLI_RENDER_REMOTE_API_KEY`), never from the command line.

`configure` writes to the project `.usd-cli/config.toml` by default (`--global` for
`~/.config/usd-cli`), then probes `/health` so a bad URL or a still-warming GPU is reported
immediately. Use `--no-test` to skip the probe.

Or set it by hand:

```toml
# .usd-cli/config.toml
[render]
renderer = "remote"
remote_url = "https://gpu-host"
remote_api_key = "" # prefer USD_CLI_RENDER_REMOTE_API_KEY secret injection
```

or per-invocation:

```bash
USD_CLI_RENDER_RENDERER=remote USD_CLI_RENDER_REMOTE_URL=https://gpu-host \
USD_CLI_RENDER_REMOTE_API_KEY="$OVRTX_API_KEY" usd-cli render --res 1024x1024
```

**Concurrency:** rendering is serial. A caller may wait briefly for the active job; excess
requests receive 429. For parallel rendering, run one isolated service per GPU behind an
authenticated scheduler or load balancer.

## Environment knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `PORT` | `8000` | Listen port |
| `OVRTX_API_KEY` | required | Bearer authentication key |
| `OVRTX_MAX_BODY_BYTES` | `73400320` | HTTP request body bound (staged/decompressed scenes are bounded at 8× this, advertised on `/live` as `max_scene_bytes`) |
| `OVRTX_CAS_DIR` | `<tmp>/ovrtx_cas` | Content-addressed blob store root (CAS transport) |
| `OVRTX_CAS_MAX_BYTES` | `21474836480` | Blob store quota; LRU-evicted, 24h TTL |
| `OVRTX_REQUEST_TIMEOUT` | `1800` | API-side render timeout (s) |
| `OVRTX_LOG_LEVEL` | `warn` | OVRTX log verbosity |
| `OVRTX_NUM_SENSOR_UPDATES` | `64` | Path-tracer step iterations (quality knob) |
| `OVRTX_RENDER_MODE` | `` | `` = derive from request mode (`fast`→rt1, `quality`→rt2); or pin `rt1`/`rt2`/`pt` |
| `OVRTX_DEFAULT_HDRI_INTENSITY` | `600` bundled / `1` custom | Intensity of the lightless-scene dome; set this with a custom HDRI when its exposure needs a value other than 1 |
| `WU_OVRTX_DEFAULT_HDRI` | bundled `studio.exr` | Optional absolute path to a replacement lat-long HDRI; without an intensity override a custom map uses intensity 1 |
| `OVRTX_DAEMON_START_TIMEOUT` | `600` | Daemon boot timeout (s) |
| `OVRTX_DAEMON_RENDER_TIMEOUT` | `1800` | Per-render timeout (s) |
| `WU_OVRTX_VENV_DIR` | `~/.cache/usd-cli/ovrtx_venv` | Pre-baked ovrtx venv path |
