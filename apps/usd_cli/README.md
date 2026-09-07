# usd-cli component

`usd-cli` is a low-level component of NVIDIA Content Agents. It is not a standalone repository, product, workflow,
or independently released application.

The component exposes selected OpenUSD and Omniverse capabilities through a compact
command-line interface and a project-local daemon so agents can inspect and modify USD
stages without generating large Python programs or dumping entire stages into context.

## Role in Content Agents

Content Agents workflows own task interpretation, policy, sequencing, recovery,
validation, and final artifacts. `usd-cli` only provides explicit scene operations to
those workflows:

- compact scene inspection and stable references;
- bounded USD edits and structural diffs;
- material and physics-schema operations;
- OVRTX-backed rendering through local or managed remote infrastructure;
- raw observations that a calling workflow may use as evidence.

The component does not choose workflows, make content decisions, define acceptance
criteria, or turn low-level observations into workflow-level verdicts. Those responsibilities
remain in `content_agent_workflows`, `content-workflow-cli`, and the applicable
`content-workflow-*` skill.

## Installation in the repository

Install `usd-cli` only from a Content Agents checkout. From the repository root:

```bash
uv pip install -e "apps/usd_cli[cli,server]" \
  --overrides apps/usd_cli/requirements/usd-exchange-override.txt
usd-cli --version
```

The override keeps `usd-exchange` as the single provider of the native `pxr` modules in
the shared Content Agents environment. Do not add a second OpenUSD provider to that
environment.

For repository setup on native Linux or WSL2, prefer
`scripts/setup_content_agent.sh`, which installs the component with the rest of
the Content Agents stack. Native Windows execution is unsupported in the 0.6
release. On a Windows host, run the supported workflow inside WSL2. The
`scripts/setup_content_agent.ps1` helper is retained for development and future
qualification only.

## Usage

Workflow code should invoke the package-owned executable resolved from this checkout.
Do not authorize an unrelated `usd-cli` found on `PATH`.

Use the repository-owned skill for low-level operating guidance:

- [usd-cli skill](../../agentic/.agents/skills/usd-cli/SKILL.md)
- [generated command reference](docs/cli-reference.md)
- `usd-cli --help` and `usd-cli <command> --help`

Requests for material authoring, physics workflows, scene decomposition, conversion,
large-scene processing, or SimReady preparation must be routed through the appropriate
Content Agents workflow. Direct use of `usd-cli` is reserved for explicit low-level USD
inspection, editing, rendering, or validation.

## Component layout

| Path | Responsibility |
|---|---|
| `src/usd_cli/` | Thin command-line client |
| `src/usd_server/` | Authenticated project-local daemon |
| `src/usd_core/` | OpenUSD and Omniverse operation adapters |
| `src/usd_telemetry/` | Explicit opt-in invocation telemetry wrapper |
| `apps/ovrtx_rendering_api/` | Managed remote OVRTX adapter used by Content Agents infrastructure |
| `docs/cli-reference.md` | Generated command reference |

Design notes, benchmarks, notebooks, example-task harnesses, and engineering investigations
are internal engineering material. They are not part of the public Content Agents
documentation surface.

## Development

Run component checks from `apps/usd_cli` inside the Content Agents checkout:

```bash
make quality
make test
make check-docs
```

The repository-level CI workflow at `.github/workflows/usd-cli-component-gate.yml` owns
component verification. Repository-level contribution, security, licensing, and delivery
policies apply; this directory intentionally carries no independent roadmap, changelog,
security policy, contributing guide, third-party notice, code-owner policy, or product
lifecycle.

See the Content Agents [license](../../LICENSE),
[third-party notice](../../THIRD_PARTY_NOTICE.md), and
[security policy](../../SECURITY.md).
