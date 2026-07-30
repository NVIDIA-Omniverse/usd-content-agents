# Quickstart: Run the Material Agent on SimReady Demo Assets

Download a ready-made SimReady asset from the public NVIDIA catalogs and run
the material agent pipeline on it end-to-end. Works without any proprietary
data — uses only public USD assets and public VLM backends.

This reference is shared with the `physics-agent-cli` and `texture-agent-cli` skills;
the variant table below picks the right USD flavor for each agent.

## Prerequisites

Beyond the standard `material-agent` prereqs:

- **`hf` CLI** (HuggingFace Hub client): `pip install -U huggingface_hub`
- **`git`** with support for `--filter=blob:none --sparse` (git 2.25+)
- ~100 MB free disk space for the demo assets

## Default asset location

Downloaded demo assets live outside the repo so they persist across clones:

```
~/content-agents-data/simready/
├── hf/      # assets from nvidia/PhysicalAI-SimReady-Warehouse-01
├── github/  # sparse-clone of NVIDIA/simready-foundation
└── configs/ # per-asset quickstart configs (optional location)
```

The `~` above is a shell/filesystem shorthand — your shell expands it when
running the `hf download` / `git clone` commands below. When you later put
this path into a YAML config file (Step 2), use the expanded absolute form
(`/home/<USER>/...`) because the config loader does not perform tilde
expansion.

Override with `CONTENT_AGENTS_DATA=/some/other/path` if desired; substitute
that path everywhere below.

## Curated demo assets

Seven assets cover prop-style and robot-style content, with existing
material bindings that the agent can re-predict.

### HuggingFace (`nvidia/PhysicalAI-SimReady-Warehouse-01`)

Full-resolution props with MDL materials and textures.

| Asset | HF path | Size |
|---|---|---|
| Steel rolling scaffold | `Props/general/SM_SteelRollingScaffold_A01_01` | ~36 MB |
| Cleaning trolley | `Props/general/SM_CleaningTrolley_B01_01` | ~20 MB |

(The aluminum step ladder from the same dataset is not listed here — the
repo already ships a public ladder example at
`apps/material_agent/data/examples/ladder/`, driven by
`apps/material_agent/configs/unified_example.yaml`. Run that first if you
want the zero-setup demo; use the assets above when you want to try
something new.)

Each folder contains:
- `<Name>.usd` (entry, material-focused)
- `<Name>_physics.usd` (physics variant)
- `materials/*.mdl` + `materials/textures/<name>/*`
- `.thumbs/256x256/<Name>.usd.png` — usable as a reference image

### GitHub (`NVIDIA/simready-foundation`, `sample_content/`)

Prop + robot assets with multiple USD variants.

| Asset | GitHub path (variant used for material agent) |
|---|---|
| Electricians toolbox | `sample_content/common_assets/props_general/obs_electricians_large_tool_box_a01/simready_usd/` |
| UR10 robot arm | `sample_content/common_assets/robots_general/ur10/simready_usd/` |

Note: The UR10 uses USD payloads and instanced geometry. Set
`steps.build_dataset_usd.prim_filters.skip_instances: false` in the config,
otherwise the agent sees zero processable meshes.

## Agent-to-variant mapping

| Agent | HF entry USD | GitHub variant |
|---|---|---|
| **material-agent** | `<Name>.usd` | `simready_usd/` |
| **physics-agent** | `<Name>_physics.usd` | `simready_physx_usd/` |
| **texture-agent** | `<Name>.usd` (needs existing materials) | `simready_usd/` |

## Step 1 — Download the assets

Pick one asset to start with. Commands below download the steel rolling
scaffold (HF) and the electricians toolbox (GitHub). Tested; runs in under a
minute on a warm network.

### HuggingFace prop

```bash
mkdir -p ~/content-agents-data/simready/hf
hf download \
  --repo-type dataset nvidia/PhysicalAI-SimReady-Warehouse-01 \
  --include "Props/general/SM_SteelRollingScaffold_A01_01/*" \
  --local-dir ~/content-agents-data/simready/hf/
```

Repeat with a different `--include` pattern for other props. You can pass
multiple `--include` flags to the same invocation.

### GitHub asset (sparse clone)

```bash
mkdir -p ~/content-agents-data/simready/github
cd ~/content-agents-data/simready/github

git clone --filter=blob:none --sparse --depth 1 \
  https://github.com/NVIDIA/simready-foundation.git

cd simready-foundation
git sparse-checkout set \
  'sample_content/common_assets/props_general/obs_electricians_large_tool_box_a01/simready_usd' \
  'sample_content/common_assets/robots_general/ur10/simready_usd'
```

To add another asset later, re-run `git sparse-checkout set` with the full
list of paths you want, or use `git sparse-checkout add <path>`.

## Step 2 — Copy and edit the example config

`apps/material_agent/configs/unified_example.yaml` already uses the public
`nim` backend and remote rendering. Copy it and edit a few paths:

```bash
cp apps/material_agent/configs/unified_example.yaml \
   apps/material_agent/configs/simready_scaffold.yaml
```

Then in the copy, change only these fields (use absolute paths — `~` is not
expanded by the config loader):

| Field | New value |
|---|---|
| `project.name` / `project.session_id` | `simready_scaffold` |
| `input.usd_path` | `/home/<USER>/content-agents-data/simready/hf/Props/general/SM_SteelRollingScaffold_A01_01/SM_SteelRollingScaffold_A01_01.usd` |
| `input.reference_images` | `/home/<USER>/content-agents-data/simready/hf/Props/general/SM_SteelRollingScaffold_A01_01/.thumbs/256x256/SM_SteelRollingScaffold_A01_01.usd.png` |

For **UR10** only, also add `skip_instances: false` under
`steps.build_dataset_usd.prim_filters` (the default is `true` and would
filter out all its instanced meshes).

`materials.path` in the example already points at the shipped default
library — no change needed.

## Step 2.5 — Verify your VLM key

The pipeline calls the VLM on step 5 of 8, so an invalid key silently wastes
the first four steps. Probe it first with `wu chat` — same backend-resolution
as the pipeline's predict step:

```bash
wu chat "hi" --backend nim --model google/gemma-4-31b-it --max-tokens 8
```

If you get a response, the key works. If you get `403 Forbidden`, the key is
not scoped for build.nvidia.com — generate one at https://build.nvidia.com.
Other NVIDIA keys may not be interchangeable.

If you changed `predict.vlm.backend` to another public provider, swap the
flag (`wu chat` honors any registered backend):

```bash
wu chat "hi" --backend openai    --model gpt-4o                   --max-tokens 8
wu chat "hi" --backend anthropic --model claude-sonnet-4-20250514 --max-tokens 8
wu chat "hi" --backend gemini    --model gemini-2.5-pro           --max-tokens 8
```

Running the docker-compose `--profile vlm` sidecar? `MA_VLM_NIM_BASE_URL`
redirects `wu chat --backend nim` to the local container automatically —
the same probe still works.

## Step 3 — Run the pipeline

```bash
# From the content-agents repo root, with venv activated
material-agent run apps/material_agent/configs/simready_scaffold.yaml --dry-run

# Real run (uses VLM credits and remote rendering)
material-agent run apps/material_agent/configs/simready_scaffold.yaml
```

Dry-run prints the execution plan and exits without calling VLMs or
renderers. Use it first to confirm paths resolve and steps are wired.

## Troubleshooting

- **`hf: command not found`** — `pip install -U huggingface_hub`
  (the newer `hf` binary replaces the legacy `huggingface-cli`).
- **GitHub sparse checkout pulls everything** — confirm git version is 2.25+
  and that `--filter=blob:none --sparse` was on the initial `git clone`.
- **UR10 pipeline finds 0 meshes** — set `skip_instances: false` under
  `build_dataset_usd.prim_filters`. UR10 uses instanced geometry that would
  otherwise be filtered out.
- **"No module named world_understanding.cli"** — editable install points
  at a deleted path. Re-run `uv pip install -e .` from the repo root.
- **Paths not resolving** — all relative paths are relative to the config
  file's directory, and `~` is not expanded. Use absolute paths or adjust
  the config location.
