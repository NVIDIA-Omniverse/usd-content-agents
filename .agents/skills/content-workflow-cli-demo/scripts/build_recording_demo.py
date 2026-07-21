#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a product-demo plan for a content-workflow-cli workflow."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

DEFAULT_USD = "apps/material_agent/data/examples/ladder/sources/usd/ladder.usd"
DEFAULT_REFERENCE = (
    "apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg"
)
DEFAULT_MATERIALS = (
    "apps/material_agent/data/materials/material_libs_default/materials.yaml"
)
DEFAULT_OUTPUT_DIR = ".local-runs/content-workflow-cli/ladder-product-demo"
DEFAULT_OUTPUT_USD = f"{DEFAULT_OUTPUT_DIR}/ladder_material_assignments.usda"
DEFAULT_PROMPT = (
    "Apply materials matching the reference ladder: brushed aluminum rails and "
    "steps, blue molded-plastic top and tray, and black rubber feet. Use Content "
    "Authoring Tool preview, render verification, and the material apply API "
    "before finalizing."
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a storyboard and exact command for a "
            "content-workflow-cli material assignment product demo."
        )
    )
    parser.add_argument("--usd", default=DEFAULT_USD, help="USD asset path.")
    parser.add_argument(
        "--reference-image",
        action="append",
        default=[],
        help="Reference image path. May be repeated.",
    )
    parser.add_argument(
        "--materials-yaml",
        default=DEFAULT_MATERIALS,
        help="Material library YAML path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="content-workflow-cli workflow run directory.",
    )
    parser.add_argument(
        "--output-usd",
        default=None,
        help=(
            "Durable material apply output USD. Defaults to "
            "<output-dir>/<input-stem>_material_assignments.usda."
        ),
    )
    parser.add_argument(
        "--workbench-url",
        default="http://127.0.0.1:8088",
        help="Content Authoring Tool endpoint to show during the demo.",
    )
    parser.add_argument(
        "--runner",
        choices=["codex", "claude"],
        default="codex",
        help="Child agent runner.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Short extra instruction prompt to pass to the workflow.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Include --dry-run in the generated CLI command.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Enable Scene Optimizer in the generated workflow. The staged-checkout "
            "default is --no-optimize because optimizer resources are optional."
        ),
    )
    parser.add_argument(
        "--no-keep-workbench",
        action="store_true",
        help="Do not include --keep-workbench in the generated CLI command.",
    )
    parser.add_argument(
        "--write-dir",
        type=Path,
        default=None,
        help="Optional directory where demo_plan.md and run_demo.sh are written.",
    )
    args = parser.parse_args()
    if args.output_usd is None:
        args.output_usd = derive_output_usd(args.output_dir, args.usd)

    reference_images = args.reference_image or [DEFAULT_REFERENCE]
    command = build_command(args, reference_images)
    plan = build_plan(args, reference_images, command)

    if args.write_dir is not None:
        args.write_dir.mkdir(parents=True, exist_ok=True)
        plan_path = args.write_dir / "demo_plan.md"
        script_path = args.write_dir / "run_demo.sh"
        plan_path.write_text(plan, encoding="utf-8")
        script_path.write_text(shell_script(command, args.output_dir), encoding="utf-8")
        script_path.chmod(0o755)
        print(f"Wrote {plan_path}")
        print(f"Wrote {script_path}")
    else:
        print(plan)
    return 0


def build_command(args: argparse.Namespace, reference_images: list[str]) -> list[str]:
    command = [
        "content-workflow-cli",
        "materials",
        "assign",
        "--usd",
        args.usd,
    ]
    for reference_image in reference_images:
        command.extend(["--reference-image", reference_image])
    command.extend(
        [
            "--materials-yaml",
            args.materials_yaml,
            "--workbench-url",
            args.workbench_url,
            "--output-dir",
            args.output_dir,
            "--output-usd",
            args.output_usd,
            "--runner",
            args.runner,
            "--additional-instructions",
            args.prompt,
        ]
    )
    command.append("--optimize" if args.optimize else "--no-optimize")
    if not args.no_keep_workbench:
        command.append("--keep-workbench")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_plan(
    args: argparse.Namespace, reference_images: list[str], command: list[str]
) -> str:
    quoted_command = format_runnable_command(command, args.output_dir)
    reference_lines = "\n".join(f"  - `{path}`" for path in reference_images)
    capture_mode = "dry run" if args.dry_run else "real run"
    write_dir = (
        f"`{args.write_dir}` (contains `demo_plan.md` and `run_demo.sh` only)"
        if args.write_dir is not None
        else "not written; the plan is printed to stdout"
    )
    dry_run_note = (
        "\n- Dry-run plans are for command rehearsal only. Remove `--dry-run` "
        "before rerendering because the rerender step requires durable apply "
        "output.\n"
        if args.dry_run
        else ""
    )
    return f"""# Agentic Content Authoring Workflow Demo Plan

## Capture Mode

- Mode: {capture_mode}
- Runner: `{args.runner}`
- Content Authoring Tool URL: `{args.workbench_url}`
- Plan/launcher directory: {write_dir}
- Workflow run directory: `{args.output_dir}`
- Durable apply output USD: `{args.output_usd}`
{dry_run_note}

## Inputs

- USD: `{args.usd}`
- Materials YAML: `{args.materials_yaml}`
- Reference image(s):
{reference_lines}
- Extra instruction prompt: `{args.prompt}`

## Terminal Command

```bash
{quoted_command}
```

## Window Layout

- Left: terminal at large font running the command above.
- Right: Content Authoring Tool viewport or API UI, showing the loaded asset and
  material/render updates.
- Optional: reference image preview near the authoring tool window.

## Demo Beats

1. Show the command before pressing Enter.
2. Show run directory creation and authoring session setup.
3. Show observable trace lines for material inspection and preview overrides.
4. Show render preview and material updates in the second window.
5. Show one issue detection and correction pass if it occurs naturally.
6. End on final render artifacts and trace files:
   - `final_renders/`
   - `assignments.json`
   - `visual_quality_assessment.json`
   - `trace/operation_trace.md`

## Rerender Input

Rerender the workflow result in place; do not rename or copy the durable USD:

```bash
python3 .agents/skills/content-workflow-cli-demo/scripts/rerender_demo_assets.py \\
  --source-usd {shlex.quote(args.usd)} \\
  --assigned-usd {shlex.quote(args.output_usd)}
```

## Terminal Narration Labels

Use labels like these in spoken narration or chapter notes:

- User launches the CLI from the terminal.
- The agent opens an authoring session and inspects material candidates.
- The agent uses preview/apply APIs instead of editing USD blindly.
- The Content Authoring Tool renders evidence views for visual review.
- The agent iterates when the render exposes a material issue.
- Final artifacts are saved for audit and replay.
"""


def derive_output_usd(output_dir: str, usd: str) -> str:
    """Place the durable apply result under the canonical workflow run."""
    input_stem = Path(usd).stem or "asset"
    return str(Path(output_dir) / f"{input_stem}_material_assignments.usda")


def format_runnable_command(command: list[str], output_dir: str) -> str:
    """Create the run directory before invoking the CLI with --output-usd."""
    return f"mkdir -p -- {shlex.quote(output_dir)}\n{format_shell_command(command)}"


def shell_script(command: list[str], output_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

{format_runnable_command(command, output_dir)}
"""


def format_shell_command(command: list[str]) -> str:
    lines = [" ".join(shlex.quote(part) for part in command[:3])]
    index = 3
    while index < len(command):
        part = command[index]
        next_part = command[index + 1] if index + 1 < len(command) else None
        if (
            part.startswith("--")
            and next_part is not None
            and not next_part.startswith("--")
        ):
            lines.append(f"  {shlex.quote(part)} {shlex.quote(next_part)}")
            index += 2
        else:
            lines.append(f"  {shlex.quote(part)}")
            index += 1
    return " \\\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
