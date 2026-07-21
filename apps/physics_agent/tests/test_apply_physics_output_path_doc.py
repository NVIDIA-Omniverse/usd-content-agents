# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the documented Physics Agent output path.

The public README and `lightbulb.yaml` claim `apply_physics` writes its
output USD to `{working_dir}/physics/<input_stem>_physics<output_ext>`.
USDZ inputs intentionally default to USDA output so runtime-resolved MDL
references do not have to be bundled into a new USDZ package. Pin that
contract by running the actual `UnifiedPipelineConfigTask` against the public
`lightbulb.yaml` and reading the autowired
`step_configs["apply_physics"]["output_usd_path"]` — so a future change
to `_autowire_paths`, `STEP_OUTPUT_DIRS`, or the path resolver cannot
drift the output path silently away from the docs.
"""

from pathlib import Path

from physics_agent.config.unified_config import UnifiedPipelineConfigTask

REPO_ROOT = Path(__file__).resolve().parents[3]
LIGHTBULB_YAML = REPO_ROOT / "apps" / "physics_agent" / "configs" / "lightbulb.yaml"


def test_lightbulb_apply_physics_autowire_matches_doc() -> None:
    task = UnifiedPipelineConfigTask()
    # Force apply_physics into the executed-step set; the default lightbulb
    # config disables some upstream steps, but we want the autowire pass to
    # build a concrete output_usd_path for apply_physics so the regression
    # check has something to compare to.
    context = task.run(
        {"config_path": str(LIGHTBULB_YAML), "only_steps": ["apply_physics"]}
    )

    expected_working_dir = LIGHTBULB_YAML.parent / ".lightbulb"
    assert context["working_dir"] == expected_working_dir, (
        f"working_dir resolved to {context['working_dir']}; expected "
        f"{expected_working_dir} per the public README's 'Where outputs land'."
    )

    apply_cfg = context["step_configs"]["apply_physics"]
    actual_output = Path(apply_cfg["output_usd_path"])
    expected_output = expected_working_dir / "physics" / "light_bulb_01_physics.usda"
    assert actual_output == expected_output, (
        f"apply_physics autowired output_usd_path={actual_output}; the "
        "docs promise `{working_dir}/physics/<input_stem>_physics<output_ext>` "
        f"= {expected_output}. Drift here means the public README and "
        "lightbulb.yaml's apply_physics step comment also need to be "
        "updated."
    )
