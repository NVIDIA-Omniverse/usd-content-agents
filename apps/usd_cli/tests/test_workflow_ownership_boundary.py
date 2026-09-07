# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def test_usd_cli_does_not_import_workflow_policy_packages() -> None:
    source_root = Path(__file__).parents[1] / "src"
    forbidden = (
        "content_agent_workflows",
        "content_workflow_cli",
        "materials.yaml",
        "yaml.safe_load",
        "USD_CLI_WORKFLOW_",
        "workflow-confined",
    )

    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{path.relative_to(source_root)}: {token}")

    assert not violations, (
        "usd-cli must remain a low-level DCC-like scene tool; workflow policy, "
        "materials.yaml resolution, and acceptance semantics belong to "
        f"content-agent workflows: {violations}"
    )
