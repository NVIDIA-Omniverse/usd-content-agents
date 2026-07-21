# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = (
    REPO_ROOT / ".agents" / "skills" / "content-workflow-cli-demo" / "scripts"
)


def load_skill_script(script_name: str) -> ModuleType:
    if script_name == "rerender_demo_assets":
        pytest.importorskip(
            "pxr", reason="USD bindings are required by rerender script"
        )

    path = SCRIPTS_ROOT / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"content_workflow_cli_demo_{script_name}",
        path,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_defaults_use_public_assets_and_one_workflow_output() -> None:
    build_demo = load_skill_script("build_recording_demo")

    for relative_path in (
        build_demo.DEFAULT_USD,
        build_demo.DEFAULT_REFERENCE,
        build_demo.DEFAULT_MATERIALS,
    ):
        assert ".data" not in Path(relative_path).parts
        assert (REPO_ROOT / relative_path).is_file()

    assert build_demo.DEFAULT_OUTPUT_USD == build_demo.derive_output_usd(
        build_demo.DEFAULT_OUTPUT_DIR,
        build_demo.DEFAULT_USD,
    )


def test_build_recording_demo_command_quotes_repeated_references() -> None:
    build_demo = load_skill_script("build_recording_demo")
    args = argparse.Namespace(
        usd="assets/g1 robot.usd",
        materials_yaml="materials/default materials.yaml",
        workbench_url="http://127.0.0.1:8088",
        output_dir=".local-runs/content-workflow-cli/g1-product-demo",
        output_usd=(
            ".local-runs/content-workflow-cli/g1-product-demo/"
            "g1_material_assignments.usda"
        ),
        runner="claude",
        prompt="match dark joints and silver shell",
        no_keep_workbench=False,
        dry_run=True,
        optimize=False,
        write_dir=Path(".local-runs/content-workflow-cli-demo/g1-product-demo"),
    )

    command = build_demo.build_command(args, ["refs/front view.png", "refs/side.png"])

    assert command[:5] == [
        "content-workflow-cli",
        "materials",
        "assign",
        "--usd",
        "assets/g1 robot.usd",
    ]
    assert command.count("--reference-image") == 2
    assert command[command.index("--output-usd") + 1] == args.output_usd
    assert "--no-optimize" in command
    assert "--keep-workbench" in command
    assert command[-1] == "--dry-run"

    formatted = build_demo.format_shell_command(command)
    assert "assets/g1 robot.usd" in formatted
    assert "'refs/front view.png'" in formatted
    assert "match dark joints and silver shell" in formatted

    plan = build_demo.build_plan(args, ["refs/front view.png"], command)
    assert "Dry-run plans are for command rehearsal only" in plan
    assert f"Workflow run directory: `{args.output_dir}`" in plan
    assert f"Durable apply output USD: `{args.output_usd}`" in plan
    assert f"--assigned-usd {args.output_usd}" in plan
    assert f"mkdir -p -- {args.output_dir}" in plan

    launcher = build_demo.shell_script(command, args.output_dir)
    assert f"mkdir -p -- {args.output_dir}" in launcher
    assert launcher.index("mkdir -p") < launcher.index("content-workflow-cli")


def test_demo_default_paths_do_not_drift_between_workflow_and_rerender() -> None:
    build_demo = load_skill_script("build_recording_demo")
    render_video = load_skill_script("render_demo_video")
    rerender_assets = load_skill_script("rerender_demo_assets")

    assert rerender_assets.DEFAULT_SOURCE_USD == build_demo.DEFAULT_USD
    assert rerender_assets.DEFAULT_ASSIGNED_USD == build_demo.DEFAULT_OUTPUT_USD
    assert render_video.DEFAULT_RUN_DIR == build_demo.DEFAULT_OUTPUT_DIR
    assert (
        render_video.DEFAULT_ASSIGNED_USD_LABEL
        == Path(build_demo.DEFAULT_OUTPUT_USD).name
    )
    assert render_video.DEFAULT_USD == build_demo.DEFAULT_USD
    assert render_video.DEFAULT_REFERENCE_IMAGE == build_demo.DEFAULT_REFERENCE


def test_demo_output_usd_is_derived_from_custom_run_and_input() -> None:
    build_demo = load_skill_script("build_recording_demo")

    assert build_demo.derive_output_usd(
        ".local-runs/content-workflow-cli/custom-run",
        "assets/custom car.usdz",
    ) == (
        ".local-runs/content-workflow-cli/custom-run/"
        "custom car_material_assignments.usda"
    )


def test_render_demo_video_parses_summary_and_stats_defensively(
    tmp_path: Path,
) -> None:
    render_video = load_skill_script("render_demo_video")
    (tmp_path / "final_summary.md").write_text(
        "\n".join(
            [
                "Coverage invariant: **passed**",
                "Status: `complete`",
                "| Visible/renderable | 12/10 |",
                "| Material decision | 7 |",
                "| Preview override | 3 |",
            ]
        ),
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "codex_result.json").write_text(
        json.dumps({"usage": None}),
        encoding="utf-8",
    )
    (tmp_path / "api_operation_counts.json").write_text(
        json.dumps({"api_operation_count_total": 5}),
        encoding="utf-8",
    )
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "events.jsonl").write_text(
        '{"phase": "inspect", "summary": "Workbench preview"}\nnot json\n',
        encoding="utf-8",
    )

    summary_lines = render_video.load_summary_lines(tmp_path)
    stats_lines = render_video.load_stats_lines(
        tmp_path,
        runner_label="agent runner",
        model_label="configured model",
        effort_label="configured effort",
    )
    events = render_video.load_events(tmp_path)

    assert summary_lines == [
        "Coverage invariant: passed",
        "Status: complete",
        "12/10 visible/renderable candidates",
        "7 explicit material decisions",
        "3 preview override prims",
    ]
    assert "Input tokens: n/a" in stats_lines
    assert "Content Authoring Tool API queries: 5 total / n/a successful" in stats_lines
    assert events == [("inspect", "Workbench preview")]


def test_render_demo_video_reads_claude_usage_and_rejects_custom_canvas(
    tmp_path: Path,
) -> None:
    render_video = load_skill_script("render_demo_video")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "claude_result.json").write_text(
        json.dumps({"usage": {"input_tokens": 12, "output_tokens": 34}}),
        encoding="utf-8",
    )

    stats_lines = render_video.load_stats_lines(
        tmp_path,
        runner_label="Claude",
        model_label="configured model",
        effort_label="configured effort",
    )

    assert "Input tokens: 12" in stats_lines
    assert "Output tokens: 34" in stats_lines
    with pytest.raises(ValueError, match="fixed at 1920x1080"):
        render_video.validate_video_dimensions(1280, 720)


def test_render_demo_video_scene_text_uses_workflow_parameters(
    tmp_path: Path,
) -> None:
    render_video = load_skill_script("render_demo_video")
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"not opened by build_scenes")
    assets = dict.fromkeys(
        [
            "reference",
            "initial",
            "initial_front",
            "initial_side",
            "logo",
            "preview_front",
            "preview_side",
            "final",
            "final_front",
            "final_oblique",
        ],
        image_path,
    )

    scenes = render_video.build_scenes(
        tmp_path,
        assets,
        runner_label="Claude",
        model_label="configured model",
        effort_label="configured effort",
        usd="assets/custom_car.usd",
        reference_image=Path("refs/custom_car.png"),
        workflow_runner="claude",
        workflow_command=(
            "content-workflow-cli materials assign --usd assets/custom_car.usd\n"
            "  --runner claude"
        ),
        prompt="Paint the car red and keep the tires dark.",
        target_description="red body paint, black tires, and clear glass",
        assigned_usd_label="custom_car_materials.usda",
    )

    lines = "\n".join(line for scene in scenes for line in scene.lines)
    assert "custom_car.png" in lines
    assert "assets/custom_car.usd" in lines
    assert "Paint the car red" in lines
    assert "red body paint" in lines
    assert "custom_car_materials.usda" in lines
    assert "unitree" not in lines.lower()
    assert "g1_material_assignments" not in lines


def test_rerender_asset_names_match_video_render_asset_contract(
    tmp_path: Path,
) -> None:
    render_video = load_skill_script("render_demo_video")
    rerender_assets = load_skill_script("rerender_demo_assets")
    expected_video_assets = {
        "baseline_contact_sheet.png",
        "baseline_front_plus_y.png",
        "baseline_right_plus_x.png",
        "final_contact_sheet.png",
        "final_front_plus_x.png",
        "final_side_plus_y.png",
        "final_oblique_plus_x_minus_y_plus_z.png",
    }
    rerender_outputs = {
        spec.filename
        for spec in rerender_assets.BASELINE_RENDERS + rerender_assets.ASSIGNED_RENDERS
    } | {"baseline_contact_sheet.png", "final_contact_sheet.png"}

    assert expected_video_assets <= rerender_outputs

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    reference_image = tmp_path / "reference.png"
    reference_image.write_bytes(b"reference")
    for filename in expected_video_assets:
        (assets_dir / filename).write_bytes(filename.encode("utf-8"))

    assets = render_video.load_assets(tmp_path / "run", assets_dir, reference_image)

    assert assets["reference"] == reference_image
    assert assets["initial"] == assets_dir / "baseline_contact_sheet.png"
    assert assets["initial_front"] == assets_dir / "baseline_front_plus_y.png"
    assert assets["initial_side"] == assets_dir / "baseline_right_plus_x.png"
    assert assets["logo"] == assets_dir / "final_front_plus_x.png"
    assert assets["preview_front"] == assets_dir / "final_front_plus_x.png"
    assert assets["preview_side"] == assets_dir / "final_side_plus_y.png"
    assert assets["final"] == assets_dir / "final_contact_sheet.png"
    assert assets["final_front"] == assets_dir / "final_front_plus_x.png"
    assert (
        assets["final_oblique"]
        == assets_dir / "final_oblique_plus_x_minus_y_plus_z.png"
    )


def test_rerender_scene_overlay_paths_follow_source_asset_name(tmp_path: Path) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")

    unmaterialized_usd, clay_baseline_usd = rerender_assets.scene_overlay_paths(
        tmp_path / "scenes", Path("assets/custom_car.usd")
    )

    assert unmaterialized_usd == tmp_path / "scenes" / "custom_car_unmaterialized.usda"
    assert clay_baseline_usd == tmp_path / "scenes" / "custom_car_clay_baseline.usda"


def test_rerender_render_set_writes_outputs_and_cleans_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")
    source_image = tmp_path / "render.png"
    source_image.write_bytes(b"rendered image")
    calls: list[tuple[str, dict[str, object]]] = []
    deleted: list[tuple[str, str]] = []

    def fake_post_json(
        _base_url: str,
        path: str,
        payload: dict[str, object],
        *,
        timeout_seconds: int = 120,
    ) -> dict[str, object]:
        calls.append((path, payload | {"timeout_seconds": timeout_seconds}))
        if path == "/sessions":
            return {"session_id": "session-1"}
        return {"image_path": str(source_image)}

    monkeypatch.setattr(rerender_assets, "post_json", fake_post_json)
    monkeypatch.setattr(
        rerender_assets,
        "delete_session",
        lambda base_url, session_id: deleted.append((base_url, session_id)),
    )

    outputs = rerender_assets.render_set(
        workbench_url="http://tool",
        scene_path=tmp_path / "scene.usda",
        specs=[rerender_assets.RenderSpec("final_front_plus_x.png", "+x", "final")],
        assets_dir=tmp_path / "assets",
        responses_dir=tmp_path / "responses",
        width=320,
        height=240,
        hdri_light=600.0,
        dome_light=None,
        distant_light=None,
    )

    assert outputs == {
        "final_front_plus_x.png": tmp_path / "assets" / "final_front_plus_x.png"
    }
    assert outputs["final_front_plus_x.png"].read_bytes() == b"rendered image"
    assert json.loads((tmp_path / "responses" / "final_front_plus_x.json").read_text())
    assert calls[0][0] == "/sessions"
    assert calls[1][0] == "/sessions/session-1/render"
    assert deleted == [("http://tool", "session-1")]


def test_rerender_render_set_validates_missing_render_image_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")
    deleted: list[tuple[str, str]] = []

    def fake_post_json(
        _base_url: str,
        path: str,
        _payload: dict[str, object],
        *,
        timeout_seconds: int = 120,
    ) -> dict[str, object]:
        if path == "/sessions":
            return {"session_id": "session-1"}
        return {"image_path": str(tmp_path / "missing.png")}

    monkeypatch.setattr(rerender_assets, "post_json", fake_post_json)
    monkeypatch.setattr(
        rerender_assets,
        "delete_session",
        lambda base_url, session_id: deleted.append((base_url, session_id)),
    )

    with pytest.raises(RuntimeError, match="image_path does not exist"):
        rerender_assets.render_set(
            workbench_url="http://tool",
            scene_path=tmp_path / "scene.usda",
            specs=[rerender_assets.RenderSpec("final_front_plus_x.png", "+x", "final")],
            assets_dir=tmp_path / "assets",
            responses_dir=tmp_path / "responses",
            width=320,
            height=240,
            hdri_light=600.0,
            dome_light=None,
            distant_light=None,
        )

    assert deleted == [("http://tool", "session-1")]


def test_rerender_clay_baseline_binds_all_gprims(tmp_path: Path) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    source_usd = tmp_path / "source.usda"
    clay_usd = tmp_path / "clay.usda"
    stage = Usd.Stage.CreateNew(str(source_usd))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    cylinder = UsdGeom.Cylinder.Define(stage, "/World/Cylinder")
    stage.GetRootLayer().Save()

    rerender_assets.write_clay_baseline_overlay(source_usd, clay_usd)

    clay_stage = Usd.Stage.Open(str(clay_usd))
    assert clay_stage is not None
    for path in (sphere.GetPath(), cylinder.GetPath()):
        material, _relationship = UsdShade.MaterialBindingAPI(
            clay_stage.GetPrimAtPath(path)
        ).ComputeBoundMaterial()
        assert material


def test_rerender_unmaterialized_overlay_blocks_materials_and_display_colors(
    tmp_path: Path,
) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")
    Gf = pytest.importorskip("pxr.Gf")
    Sdf = pytest.importorskip("pxr.Sdf")
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    source_usd = tmp_path / "source.usda"
    unmaterialized_usd = tmp_path / "unmaterialized.usda"
    stage = Usd.Stage.CreateNew(str(source_usd))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    material = UsdShade.Material.Define(stage, "/Looks/Yellow")
    UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
    display_color = sphere.GetDisplayColorAttr()
    display_color.Set([Gf.Vec3f(1.0, 0.8, 0.0)], Usd.TimeCode(1.0))
    sphere.GetPrim().CreateAttribute(
        "primvars:displayOpacity",
        Sdf.ValueTypeNames.FloatArray,
        custom=True,
    ).Set([0.5])
    stage.GetRootLayer().Save()

    counts = rerender_assets.write_unmaterialized_overlay(
        source_usd,
        unmaterialized_usd,
    )

    assert counts == {"material_bindings": 1, "display_colors": 2, "instances": 0}
    clean_stage = Usd.Stage.Open(str(unmaterialized_usd))
    assert clean_stage is not None
    clean_sphere = clean_stage.GetPrimAtPath("/World/Sphere")
    material, _relationship = UsdShade.MaterialBindingAPI(
        clean_sphere
    ).ComputeBoundMaterial()
    assert not material
    for attr_name in rerender_assets.DISPLAY_COLOR_ATTRS:
        attr = clean_sphere.GetAttribute(attr_name)
        assert not attr or (attr.Get() is None and attr.GetNumTimeSamples() == 0)


def test_rerender_render_set_validates_missing_render_image_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")
    deleted: list[tuple[str, str]] = []

    def fake_post_json(
        _base_url: str,
        path: str,
        _payload: dict[str, object],
        *,
        timeout_seconds: int = 120,
    ) -> dict[str, object]:
        if path == "/sessions":
            return {"session_id": "session-1"}
        return {}

    monkeypatch.setattr(rerender_assets, "post_json", fake_post_json)
    monkeypatch.setattr(
        rerender_assets,
        "delete_session",
        lambda base_url, session_id: deleted.append((base_url, session_id)),
    )

    with pytest.raises(RuntimeError, match="image_path"):
        rerender_assets.render_set(
            workbench_url="http://tool",
            scene_path=tmp_path / "scene.usda",
            specs=[rerender_assets.RenderSpec("final_front_plus_x.png", "+x", "final")],
            assets_dir=tmp_path / "assets",
            responses_dir=tmp_path / "responses",
            width=320,
            height=240,
            hdri_light=600.0,
            dome_light=None,
            distant_light=None,
        )

    assert deleted == [("http://tool", "session-1")]


def test_rerender_post_json_reports_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rerender_assets = load_skill_script("rerender_demo_assets")

    class FakeResponse:
        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"not json"

    def fake_urlopen(
        _request: object,
        *,
        timeout: int,
    ) -> FakeResponse:
        return FakeResponse()

    monkeypatch.setattr(rerender_assets.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Invalid JSON response"):
        rerender_assets.post_json("http://tool", "/sessions", {"scene_path": "a.usd"})
