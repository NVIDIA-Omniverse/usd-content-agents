# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the small public workflow examples."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import trimesh
import yaml
from pxr import Sdf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = REPO_ROOT / "agentic" / "examples"
PUBLIC_WORKFLOWS = {
    "articulation",
    "cad-to-simready",
    "geometry",
    "materials",
    "mesh-segmentation",
    "physics",
    "scene",
    "simready",
    "texture",
    "validation",
}


def test_public_examples_are_grouped_by_workflow() -> None:
    assert PUBLIC_WORKFLOWS <= {
        path.name for path in EXAMPLES.iterdir() if path.is_dir()
    }
    assert not (EXAMPLES / "asset").exists()
    assert not (EXAMPLES / "convert-to-usd").exists()
    assert not (EXAMPLES / "physics-vomp").exists()


def test_public_launchers_are_executable_and_contain_no_internal_asset_urls() -> None:
    public_files = [EXAMPLES / "README.md"]
    for workflow in PUBLIC_WORKFLOWS:
        public_files.extend(
            path for path in (EXAMPLES / workflow).rglob("*") if path.is_file()
        )
    for script in (path for path in public_files if path.name.endswith(".sh")):
        assert os.access(script, os.X_OK), script
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in public_files
    )
    assert "s3://" not in text
    assert "omni-genai-dev" not in text


def test_public_bash_examples_have_native_powershell_launchers() -> None:
    for workflow in PUBLIC_WORKFLOWS:
        for bash_launcher in (EXAMPLES / workflow).rglob("run.sh"):
            powershell_launcher = bash_launcher.with_suffix(".ps1")
            assert powershell_launcher.is_file(), powershell_launcher
            powershell = powershell_launcher.read_text(encoding="utf-8")
            assert '$ErrorActionPreference = "Stop"' in powershell
            assert "@args" in powershell
            assert "$LASTEXITCODE" in powershell


def test_fused_cart_is_one_small_watertight_mesh_with_three_components() -> None:
    fixture = EXAMPLES / "mesh-segmentation" / "basic" / "fused_cart.usda"
    assert fixture.stat().st_size < 10_000
    stage = Usd.Stage.Open(str(fixture))
    assert stage is not None
    mesh = UsdGeom.Mesh.Get(stage, "/FusedCart/Geometry")
    assert mesh
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    inspected = trimesh.Trimesh(
        vertices=list(mesh.GetPointsAttr().Get()),
        faces=[indices[index : index + 3] for index in range(0, len(indices), 3)],
        process=False,
    )
    assert len(inspected.faces) == 36
    assert inspected.is_watertight
    assert len(inspected.split(only_watertight=False)) == 3


def test_small_scene_fixture_composes_pinned_simready_assets() -> None:
    fixture = EXAMPLES / "scene" / "material-pass" / "mini_workcell.usda"
    assert fixture.stat().st_size < 25_000
    layer = Sdf.Layer.FindOrOpen(str(fixture))
    assert layer is not None
    references = {
        path: layer.GetPrimAtPath(path).referenceList.prependedItems[0].assetPath
        for path in (
            "/Workcell/Workbench",
            "/Workcell/Toolbox",
            "/Workcell/Sledgehammer",
        )
    }
    assert references == {
        "/Workcell/Workbench": (
            "simready-foundation/sample_content/common_assets/props_general/"
            "obs_workbench_tool_a01/simready_usd/"
            "sm_obs_workbench_tool_a01_01.usd"
        ),
        "/Workcell/Toolbox": (
            "simready-foundation/sample_content/common_assets/props_general/"
            "obs_electricians_large_tool_box_a01/simready_usd/"
            "sm_obs_electricians_large_tool_box_a01_01.usd"
        ),
        "/Workcell/Sledgehammer": (
            "simready-foundation/sample_content/common_assets/props_general/"
            "obs_small_sledge_hammer_a01/simready_usd/"
            "sm_obs_small_sledge_hammer_a01_01.usd"
        ),
    }


def test_scene_material_pass_uses_a_pinned_public_download() -> None:
    example = EXAMPLES / "scene" / "material-pass"
    run = (example / "run.sh").read_text(encoding="utf-8")
    for fetch_name in ("fetch.sh", "fetch.ps1"):
        fetch = (example / fetch_name).read_text(encoding="utf-8")
        assert "0ed0dfbc539c9de99289771bd6848effe3ef5779" in fetch
        assert "media.githubusercontent.com/media/NVIDIA/simready-foundation" in fetch
        assert "assets.sha256" in fetch

    manifest_entries = [
        line.split(maxsplit=1)
        for line in (example / "assets.sha256").read_text().splitlines()
        if line.strip()
    ]
    manifest_paths = {path for _, path in manifest_entries}
    assert len(manifest_paths) == 21
    assert {
        "simready-foundation/sample_content/common_assets/props_general/"
        "obs_workbench_tool_a01/simready_usd/sm_obs_workbench_tool_a01_01.usd",
        "simready-foundation/sample_content/common_assets/props_general/"
        "obs_electricians_large_tool_box_a01/simready_usd/"
        "sm_obs_electricians_large_tool_box_a01_01.usd",
        "simready-foundation/sample_content/common_assets/props_general/"
        "obs_small_sledge_hammer_a01/simready_usd/"
        "sm_obs_small_sledge_hammer_a01_01.usd",
    } <= manifest_paths
    assert run.count("content-workflow-cli scene run") == 1
    assert ".data/examples/scene-material-pass/mini_workcell.usda" in run


def test_articulation_fixture_has_one_frame_and_one_drawer() -> None:
    fixture = EXAMPLES / "articulation" / "drawer" / "drawer.usda"
    assert fixture.stat().st_size < 10_000
    stage = Usd.Stage.Open(str(fixture))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/MiniCabinet"
    assert stage.GetPrimAtPath("/MiniCabinet/Frame")
    assert stage.GetPrimAtPath("/MiniCabinet/Drawer")
    cube_paths = {
        str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Cube)
    }
    assert cube_paths == {
        "/MiniCabinet/Drawer/DrawerBody",
        "/MiniCabinet/Drawer/DrawerFront",
        "/MiniCabinet/Frame/BackPanel",
        "/MiniCabinet/Frame/BottomPanel",
        "/MiniCabinet/Frame/LeftPanel",
        "/MiniCabinet/Frame/RightPanel",
        "/MiniCabinet/Frame/TopPanel",
    }


def test_drawer_launchers_freeze_the_exact_articulation_scope() -> None:
    example = EXAMPLES / "articulation" / "drawer"
    required = {
        "articulation.preparation-publisher.v1",
        "articulation.author.v1",
        "articulation.evidence.v1",
        "articulation.review.v1",
        "articulation.publish.v1",
        "validation.canonical-ovrtx-evidence.v1",
    }
    terminals = {
        "articulation.publish.v1",
        "validation.canonical-ovrtx-evidence.v1",
    }
    dependencies = {
        "articulation.author.v1=articulation.preparation-publisher.v1",
        "articulation.evidence.v1=articulation.author.v1",
        "articulation.review.v1=articulation.evidence.v1",
        "articulation.publish.v1=articulation.review.v1",
        "validation.canonical-ovrtx-evidence.v1=articulation.author.v1",
    }
    for launcher in (example / "run.sh", example / "run.ps1"):
        text = launcher.read_text(encoding="utf-8")
        assert "--exact-leaf-scope" in text
        assert "articulation.proposal-provider.v1" not in text
        selected = {
            line.split("--required-leaf ", 1)[1].split()[0]
            for line in text.splitlines()
            if "--required-leaf " in line
        }
        assert selected == required
        selected_terminals = {
            line.split("--required-terminal-leaf ", 1)[1].split()[0]
            for line in text.splitlines()
            if "--required-terminal-leaf " in line
        }
        selected_dependencies = {
            line.split("--required-leaf-dependency ", 1)[1].split()[0]
            for line in text.splitlines()
            if "--required-leaf-dependency " in line
        }
        assert selected_terminals == terminals
        assert selected_dependencies == dependencies


def test_examples_reuse_existing_package_assets() -> None:
    launchers = {
        "materials": (
            EXAMPLES / "materials" / "basic-assignment" / "run.sh",
            "apps/material_agent/data/examples/ladder",
        ),
        "physics": (
            EXAMPLES / "physics" / "basic" / "run.sh",
            "apps/physics_agent/data/examples/Lightbulb01",
        ),
        "simready": (
            EXAMPLES / "simready" / "validate-and-conform" / "run.sh",
            "apps/usd_cli/sample_assets/simready/Cube",
        ),
        "texture": (
            EXAMPLES / "texture" / "basic" / "run.sh",
            "apps/texture_agent/data/examples/ladder",
        ),
    }
    for _workflow, (launcher, asset_path) in launchers.items():
        assert asset_path in launcher.read_text(encoding="utf-8")


def test_tire_bounce_examples_use_the_agentic_tuning_contract() -> None:
    example = EXAMPLES / "physics" / "tire-bounce"
    readme = (example / "README.md").read_text(encoding="utf-8")
    tune = (example / "run-tune.sh").read_text(encoding="utf-8")
    refine = (example / "run-refine.sh").read_text(encoding="utf-8")
    scenario = yaml.safe_load((example / "scenario.yaml").read_text(encoding="utf-8"))

    for launcher in (tune, refine):
        assert launcher.count("content-workflow-cli physics apply") == 1
        assert "apps/physics_agent/data/examples/Tire_B01/tire.usdc" in launcher
        assert "--tune-engine ovphysx" in launcher
        assert "--optimizer botorch" in launcher
        assert "--max-trials 30" in launcher
        assert "--max-iterations 12" in launcher
        assert "--collision-approximation convexDecomposition" in launcher
        assert "physics-agent refine" not in launcher

    assert "--tune" in tune
    assert "--refine" not in tune
    assert "--refine" in refine
    assert "examples/physics/tire-bounce/scenario.yaml" in refine
    assert "--reference-video" not in refine
    assert "reference_media" not in refine
    assert ".mov" not in refine
    assert scenario["name"] == "drop_settle"
    assert scenario["metric"] == "max_bounce_height"
    assert scenario["target"]["drop_height_m"] == 1.0
    assert [parameter["name"] for parameter in scenario["parameters"]] == [
        "mass_scale",
        "static_friction",
        "dynamic_friction",
        "restitution",
    ]
    assert all(set(parameter) == {"name"} for parameter in scenario["parameters"])
    assert "run this from the repository root" in readme
    assert 'uv pip install -e "apps/usd_cli[cli,server]"' in readme
    assert "--overrides apps/usd_cli/requirements/usd-exchange-override.txt" in readme
    assert "../apps/usd_cli" not in readme


def test_container_slide_example_uses_the_agentic_refine_contract() -> None:
    example = EXAMPLES / "physics" / "container-slide"
    guidance = (example / "physics_guidance.md").read_text(encoding="utf-8")
    launcher = (example / "run-refine.sh").read_text(encoding="utf-8")
    scenario_text = (example / "scenario.yaml").read_text(encoding="utf-8")
    scenario = yaml.safe_load(scenario_text)

    assert launcher.count("content-workflow-cli physics apply") == 1
    assert (
        "apps/physics_agent/data/examples/Container_Gray_C04/container.usdc" in launcher
    )
    assert "examples/physics/container-slide/scenario.yaml" in launcher
    assert "--refine" in launcher
    assert "--tune-engine ovphysx" in launcher
    assert "--optimizer botorch" in launcher
    assert "--max-trials 8" in launcher
    assert "--max-iterations 3" in launcher
    assert "--collision-approximation convexHull" in launcher
    assert "material-agent" not in launcher
    assert "physics-agent refine" not in launcher

    assert scenario["name"] == "freeform"
    assert scenario["metric"] == "judge_score"
    assert scenario["target"]["initial_velocity"] == [0.8, 0.0, 0.0]
    assert scenario["parameters"] == [
        {"name": "dynamic_friction", "min": 0.08, "max": 0.6}
    ]
    assert "blue" not in scenario_text.lower()
    assert "pre-tuning visual pass" in guidance
    assert "until a tuning sweep applies" in guidance


def test_mesh_segmentation_example_opts_into_configured_codex_auth() -> None:
    launcher = EXAMPLES / "mesh-segmentation" / "basic" / "run.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "--allow-codex-configured-auth" in text
    assert "--no-memory" in text


def test_texture_example_uses_the_outer_companion_workflow() -> None:
    example = EXAMPLES / "texture" / "basic"
    launcher = (example / "run.sh").read_text(encoding="utf-8")
    powershell = (example / "run.ps1").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert launcher.count("content-workflow-cli texture run") == 1
    assert "ladder_uv_ready.usd" in launcher
    assert "--unit-action" in launcher
    assert "=generate" in launcher
    assert "M_AluminumStepLadder_B01_Plastic2" in launcher
    assert "high-visibility safety orange" in launcher
    assert "smooth" in launcher
    assert "even single-color matte finish" in launcher
    assert "requested_appearance=" in launcher
    assert '--prompt "$requested_appearance"' in launcher
    assert '--unit-appearance "$grip_prim=$requested_appearance"' in launcher
    assert "--runner codex" in launcher
    assert "texture_companion_generation_handoff.json" in launcher
    assert "content-workflow-cli texture resume" in launcher
    assert r"--run-dir \"$output_dir\"" in launcher
    assert "texture_companion_generation_handoff.json" in readme
    assert "content-workflow-cli texture resume" in readme
    assert "--runner claude" in readme
    assert "--claude-execution-mode cli" in readme
    assert "--texture-agent-url" in readme
    assert 'run_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-runs/texture-ladder}"' in readme
    assert '--run-dir "$run_dir"' in readme
    assert "$env:CONTENT_WORKFLOW_OUTPUT_DIR" in readme
    assert "--run-dir $runDir" in readme
    assert "## Expected outcome" in readme
    assert "## What changes" in readme
    assert "top plastic grip/tray assembly" in readme
    assert "dark blue" in readme
    assert "safety orange" in readme
    assert "same appearance text" in readme
    assert 'workflow_result.json` has `status: "published"' in readme
    assert "published/textured_asset.usdz" in readme
    assert "texture_terminal_receipt.json" in readme
    assert "readback/texture_saved_stage_readback.json" in readme
    assert "evidence/texture_agentic_evidence.json" in readme
    assert "repo_root=" in launcher
    assert 'if [[ "$output_dir" != /* ]]' in launcher
    assert 'output_dir="$repo_root/$output_dir"' in launcher
    assert "../runs/texture-ladder" not in launcher
    assert "--execution-mode fixed" not in launcher
    assert "--texture-agent-url" not in launcher
    assert "--texture-backend" not in launcher
    for token in (
        "content-workflow-cli texture run",
        "ladder_uv_ready.usd",
        "--unit-action",
        "=generate",
        "M_AluminumStepLadder_B01_Plastic2",
        "high-visibility safety orange",
        "smooth",
        "even single-color matte finish",
        "$requestedAppearance",
        "--runner codex",
        "texture_companion_generation_handoff.json",
        "$env:CONTENT_WORKFLOW_OUTPUT_DIR",
        "$PSNativeCommandUseErrorActionPreference = $false",
        "$PSNativeCommandUseErrorActionPreference = $previousNativePreference",
        '--run-dir `"$outputDir`"',
    ):
        assert token in powershell
    assert "agentic/examples/texture/basic/run.ps1" in readme


def test_articulation_example_uses_provider_optional_long_running_coordinator() -> None:
    example = EXAMPLES / "articulation" / "drawer"
    launcher = (example / "run.sh").read_text(encoding="utf-8")
    powershell = (example / "run.ps1").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert launcher.count("content-workflow-cli asset run") == 1
    assert "drawer.usda" in launcher
    assert "sole long-running coordinator" in launcher
    assert "exactly one prismatic joint" in launcher
    assert "do not select an external articulation proposal provider" in launcher
    assert "--runner codex" in launcher
    assert "--joint-config" not in launcher
    assert "content-workflow-cli articulation run" not in launcher
    assert "--execution-mode fixed" not in launcher
    assert "content-workflow-cli asset resume" in readme
    assert "--runner claude" in readme
    assert "--claude-execution-mode cli" in readme
    assert "articulation.proposal-provider.v1" in readme
    assert "does not configure or require a Joint proposal" in readme
    assert "## Expected outcome" in readme
    assert "## What changes" in readme
    assert "unarticulated components with no physics joint" in readme
    assert "joint_rigger/rigged.usdz" in readme
    assert "static post-authoring evidence" in readme
    assert "trajectory metrics" in readme
    assert "recording.usda" in readme
    assert "graph_terminal_receipt.json" in readme
    assert "terminal_validation.json" in readme
    for token in (
        "content-workflow-cli asset run",
        "drawer.usda",
        "sole long-running coordinator",
        "exactly one prismatic joint",
        "do not select an external articulation proposal provider",
        "--runner codex",
        "$env:CONTENT_WORKFLOW_OUTPUT_DIR",
    ):
        assert token in powershell
    assert "--joint-config" not in powershell
    assert "content-workflow-cli articulation run" not in powershell
    assert "preflight articulation-authoring-platform" in powershell
    assert 'if [[ "$output_dir" != /* ]]' in launcher
    assert 'output_dir="$repo_root/$output_dir"' in launcher
    assert "[System.IO.Path]::IsPathRooted" in powershell
    assert "$PSNativeCommandUseErrorActionPreference = $false" in powershell
    assert (
        "$PSNativeCommandUseErrorActionPreference = $previousNativePreference"
        in powershell
    )
    assert "$commandExitCode = $LASTEXITCODE" in powershell
    assert "exit $commandExitCode" in powershell
    assert "requires Linux, a Linux container, or" in readme
    assert "exits before creating workflow state" in readme
    assert "agentic/examples/articulation/drawer/run.ps1" in readme


@pytest.mark.parametrize(
    "relative_launcher",
    [
        Path("articulation/drawer/run.ps1"),
        Path("validation/basic/run.ps1"),
    ],
)
def test_powershell_example_preserves_nonzero_native_exit_code(
    relative_launcher: Path,
    tmp_path: Path,
) -> None:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("PowerShell is not installed")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    if os.name == "nt":
        fake_cli = fake_bin / "content-workflow-cli.cmd"
        fake_cli.write_text("@exit /b 7\r\n", encoding="utf-8")
    else:
        fake_cli = fake_bin / "content-workflow-cli"
        fake_cli.write_text("#!/usr/bin/env sh\nexit 7\n", encoding="utf-8")
        fake_cli.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CONTENT_WORKFLOW_OUTPUT_DIR"] = str(tmp_path / "run with spaces")
    process = subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", EXAMPLES / relative_launcher],
        check=False,
        env=env,
        capture_output=True,
        text=True,
    )

    assert process.returncode == 7, process.stderr


def test_validation_example_uses_non_mutating_agentic_coordinator() -> None:
    example = EXAMPLES / "validation" / "basic"
    launcher = (example / "run.sh").read_text(encoding="utf-8")
    powershell = (example / "run.ps1").read_text(encoding="utf-8")
    readme = (example / "README.md").read_text(encoding="utf-8")

    assert launcher.count("content-workflow-cli validate run") == 1
    assert "agentic/examples/geometry/quickstart/smoke_bracket.usda" in launcher
    assert "Do not modify the asset" in launcher
    assert "--runner codex" in launcher
    assert "--model" in launcher
    assert 'render_backend="${CONTENT_WORKFLOW_RENDER_BACKEND:-ovrtx}"' in launcher
    assert '--render-backend "$render_backend"' in launcher
    assert "ovrtx | remote" in launcher
    assert "--direct-executor" not in launcher
    assert 'if [[ "$output_dir" != /* ]]' in launcher
    assert 'output_dir="$repo_root/$output_dir"' in launcher
    assert "content-workflow-cli validate collect-evidence" in readme
    assert "content-workflow-cli validate assess" in readme
    assert "content-workflow-cli validate review-assessment" in readme
    assert "--runner claude" in readme
    assert "--claude-execution-mode cli" in readme
    assert "--model sonnet" in readme
    assert "CONTENT_WORKFLOW_RENDER_BACKEND=remote" in readme
    assert '$env:CONTENT_WORKFLOW_RENDER_BACKEND = "remote"' in readme
    assert "WSL2 cannot use local OVRTX" in readme
    assert (
        'run_dir="${CONTENT_WORKFLOW_OUTPUT_DIR:-runs/validation-smoke-bracket}"'
        in readme
    )
    assert '--output-dir "$run_dir"' in readme
    assert "$env:CONTENT_WORKFLOW_OUTPUT_DIR" in readme
    assert "--output-dir $runDir" in readme
    assert "safe_restart_required" in readme
    assert "## Expected outcome" in readme
    assert "## What changes" in readme
    assert "one long-running coding agent" in readme
    assert "byte-for-byte unchanged" in readme
    assert "No Validation service or advisory VLM provider" in readme
    assert '"outer-assessment.json"' in readme
    assert '"outer-review.json"' in readme
    assert "validation_coordinator_execution_receipt.json" in readme
    assert "standalone_validation_evidence.json" in readme
    assert "canonical_validation_assessment.json" in readme
    assert "validation_terminal_receipt.json" in readme
    assert '`terminal_disposition: "pass"' in readme
    assert '`review_disposition: "accept"' in readme
    assert "does not publish a modified USD" in readme
    for token in (
        "content-workflow-cli validate run",
        "agentic/examples/geometry/quickstart/smoke_bracket.usda",
        "Do not modify the asset",
        "--runner codex",
        "--model $model",
        "--render-backend $renderBackend",
        "$env:CONTENT_WORKFLOW_OUTPUT_DIR",
        "$env:CONTENT_WORKFLOW_MODEL",
        "$env:CONTENT_WORKFLOW_RENDER_BACKEND",
        "[System.IO.Path]::IsPathRooted",
    ):
        assert token in powershell
    assert "$PSNativeCommandUseErrorActionPreference = $false" in powershell
    assert (
        "$PSNativeCommandUseErrorActionPreference = $previousNativePreference"
        in powershell
    )
    assert "$commandExitCode = $LASTEXITCODE" in powershell
    assert "exit $commandExitCode" in powershell
    assert "agentic/examples/validation/basic/run.ps1" in readme


def test_asset_workflow_quickstart_uses_the_public_root_environment() -> None:
    quickstart = REPO_ROOT / "agentic" / "docs" / "asset_workflow_quickstart.md"
    text = quickstart.read_text(encoding="utf-8")

    assert "./scripts/setup_content_agent.sh" in text
    assert "source .venv/bin/activate" in text
    assert "claude auth status" in text
    assert "uv sync --project agentic" not in text
    assert "source agentic/.venv/bin/activate" not in text


def test_toycar_uses_a_pinned_download_and_canonical_workflow() -> None:
    example = EXAMPLES / "cad-to-simready" / "toycar"
    fetch = (example / "fetch.sh").read_text(encoding="utf-8")
    run = (example / "run.sh").read_text(encoding="utf-8")
    assert "0e3a605bda7c758293ab58432f1d51a2a355d47a" in fetch
    assert "01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c" in fetch
    assert run.count("content-workflow-cli cad-to-simready run") == 1
    assert "convert-to-usd" not in run
    assert not list(example.glob("*_guidance.md"))
