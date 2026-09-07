# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The 3 s settle window must reach every entry point, not just the API.

`PhysicsApplyWorkflowInput.simulation_duration_s` moved to 3.0 so drop
validation observes the full settle instead of judging a body mid-bounce.
The CLI wrapper and the Workbench request model both pass their own
defaults explicitly into that input, so each one silently overrides the
workflow default unless it carries the same value. These tests pin all of
them together: change one deliberately, and this file is the list of the
others that must move with it.
"""

from content_workflow_cli.cli import build_parser


def test_cli_duration_default_matches_workflow_default() -> None:
    args = build_parser().parse_args(
        ["physics", "apply", "--usd", "asset.usda", "--output-dir", "out"]
    )
    assert args.duration_s == 3.0


def test_apply_config_duration_default_matches_workflow_default() -> None:
    from content_workflow_cli.runner import PhysicsApplyConfig

    assert (
        PhysicsApplyConfig.__dataclass_fields__["simulation_duration_s"].default == 3.0
    )


def test_workflow_input_duration_default_is_three_seconds() -> None:
    from content_agent_workflows.physics.workflow import PhysicsApplyWorkflowInput

    assert (
        PhysicsApplyWorkflowInput.model_fields["simulation_duration_s"].default == 3.0
    )


def test_validate_physics_runtime_duration_default_is_three_seconds() -> None:
    import inspect

    from content_agent_workflows.physics.workflow import validate_physics_runtime

    parameters = inspect.signature(validate_physics_runtime).parameters
    assert parameters["duration_s"].default == 3.0
