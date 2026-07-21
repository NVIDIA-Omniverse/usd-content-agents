# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small remaining runtime edges for physics-agent coverage."""

from __future__ import annotations

import asyncio
import builtins
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from physics_agent.config.unified_config import UnifiedPipelineConfigTask
from physics_agent.config.validator import ConfigValidator
from physics_agent.functions import inference as inference_funcs
from physics_agent.recording.recorder import _clamp_fps
from physics_agent.tasks import interpret_user_prompt_tuning as interp
from physics_agent.tasks.config_apply_physics import ApplyPhysicsConfigTask
from physics_agent.tasks.config_identify_asset import IdentifyAssetConfigTask
from physics_agent.tasks.config_prepare_dataset import PrepareDatasetConfigTask
from physics_agent.tasks.config_usd_dataset import USDDatasetConfigTask
from physics_agent.tasks.identify_asset import IdentifyAssetTask
from physics_agent.tasks.prepare_dataset import PrepareDatasetTask
from physics_agent.tasks.reporting import GeneratePredictionReportTask
from physics_agent.tuning import backend as backend_mod
from physics_agent.tuning.capabilities import (
    BINDING_KIND_SIMULATOR_PARAMETER,
    BINDING_KIND_USD_ATTRIBUTE,
    BindingCapability,
)
from physics_agent.tuning.newton_simulator import NewtonSimulator
from physics_agent.tuning.scenario import parse_scenario
from physics_agent.tuning.scenario_resolution import (
    _backend_capabilities,
    _resolve_param_binding,
    _resolve_simulator_binding,
    _resolve_usd_binding,
)
from physics_agent.tuning.scenarios import drop_settle, freeform
from physics_agent.tuning.usd_inspector import (
    UsdTuningCandidate,
    UsdTuningReport,
    _attr_value,
    _has_authored_value,
)
from physics_agent.utils import format_prediction_output


class _Store:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = dict(values or {})

    def exists(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value


class _Listener:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.messages: list[str] = []

    def event(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)

    def error(self, message: str) -> None:
        self.messages.append(message)

    def debug(self, message: str) -> None:
        self.messages.append(message)


def test_interpreter_helpers_and_default_model_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = interp._drop_settle_example_for_params(("mass_scale",))
    assert example["parameters"] == [{"name": "mass_scale", "min": 0.5, "max": 2.0}]
    assert "most relevant parameter" in interp._parameter_guidance(("contact_ke",))
    assert interp._extract_json('prefix {"text": "escaped \\" quote"} suffix') == {
        "text": 'escaped " quote'
    }

    import world_understanding.functions.models.chat_models as chat_models

    monkeypatch.setattr(
        chat_models,
        "create_chat_model_from_config",
        lambda config: SimpleNamespace(model_name=config["model"]),
    )
    assert interp._resolve_default_chat_model().model_name

    monkeypatch.setattr(
        interp,
        "_resolve_default_chat_model",
        lambda: SimpleNamespace(model="default-model"),
    )
    monkeypatch.setattr(
        interp,
        "_call_llm",
        lambda *_args: json.dumps(
            {
                "name": "freeform",
                "metric": "judge_score",
                "target": {"description": "roll away"},
                "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
            }
        ),
    )
    scenario = interp.infer_scenario_from_prompt(
        "make it settle",
        chat_model=None,
        scenario_override={
            "name": "drop_settle",
            "parameters": [{"name": "restitution", "min": 0.0, "max": 1.0}],
        },
    )
    assert scenario.name == "drop_settle"
    assert scenario.params[0].name == "restitution"


def test_config_tasks_and_small_helpers_edges(tmp_path: Path) -> None:
    usd_path = tmp_path / "asset.usd"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")

    usd_config = tmp_path / "usd.yaml"
    usd_config.write_text(
        yaml.safe_dump({"usd_path": "asset.usd", "output_dir": "usd-out"}),
        encoding="utf-8",
    )
    usd_result = USDDatasetConfigTask().run({"config_path": str(usd_config)})
    assert usd_result["usd_path"] == str(usd_path.resolve())
    assert usd_result["output_dir"] == str((tmp_path / "usd-out").resolve())
    with pytest.raises(ValueError, match="No config_path"):
        USDDatasetConfigTask()._load_config({})
    with pytest.raises(FileNotFoundError):
        USDDatasetConfigTask()._load_config({"config_path": tmp_path / "missing.yaml"})

    prepare_result = PrepareDatasetConfigTask().run(
        {"config_dict": {"usd_dir": str(tmp_path), "reference_images": []}}
    )
    assert Path(prepare_result["dataset_path"]).name == "dataset"
    assert PrepareDatasetConfigTask()._resolve_path(str(usd_path), tmp_path) == usd_path
    with pytest.raises(ValueError, match="No config_path"):
        PrepareDatasetConfigTask()._load_config({})
    with pytest.raises(FileNotFoundError):
        PrepareDatasetConfigTask()._load_config(
            {"config_path": tmp_path / "missing.yaml"}
        )

    identify_result = IdentifyAssetConfigTask().run(
        {"config_dict": {"usd_path": str(usd_path)}}
    )
    assert Path(identify_result["output_dir"]).name == "identification"
    identify_config = tmp_path / "identify.yaml"
    identify_config.write_text(
        yaml.safe_dump({"usd_path": "asset.usd", "output_dir": "identify-out"}),
        encoding="utf-8",
    )
    identify_with_output = IdentifyAssetConfigTask().run(
        {"config_path": str(identify_config)}
    )
    assert identify_with_output["output_dir"] == str(
        (tmp_path / "identify-out").resolve()
    )
    assert IdentifyAssetConfigTask()._resolve_path(str(usd_path), tmp_path) == usd_path
    with pytest.raises(ValueError, match="No config_path"):
        IdentifyAssetConfigTask()._load_config({})
    with pytest.raises(FileNotFoundError):
        IdentifyAssetConfigTask()._load_config(
            {"config_path": tmp_path / "missing.yaml"}
        )

    empty_apply = tmp_path / "empty-apply.yaml"
    empty_apply.write_text("", encoding="utf-8")
    assert ApplyPhysicsConfigTask()._load_config({"config_path": empty_apply}) == {}
    assert format_prediction_output({"id": "a", "confidence": 0.7})["confidence"] == 0.7
    assert _clamp_fps(10_000) < 10_000


def test_prepare_dataset_remaining_skip_and_context_edges(tmp_path: Path) -> None:
    usd_dir = tmp_path / "usd"
    model_dir = usd_dir / "model"
    model_dir.mkdir(parents=True)
    dataset_path = tmp_path / "dataset"
    assignments = tmp_path / "assignments.json"
    assignments.write_text(
        json.dumps(
            {
                "assignments": {
                    "/World/StringAssigned": "segment-name",
                    "/World/DictAssigned": {"component_name": "dict-name"},
                }
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "dataset.json").write_text(
        json.dumps({"statistics": {"total_prims": 2}}),
        encoding="utf-8",
    )
    (model_dir / "prims.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prim_path": "/World/GeoOnly",
                        "world_bbox_meters": {"size": [1.0, 2.0, 3.0]},
                        "relative_metrics": {"relative_size": [0.1, 0.2, 0.3]},
                        "renders": [{"path": "skip.png", "render_mode": "depth"}],
                    }
                ),
                json.dumps(
                    {
                        "prim_path": "/World/StringAssigned",
                        "renders": [{"path": "skip2.png", "render_mode": "depth"}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    listener = _Listener()
    result = PrepareDatasetTask().run(
        {
            "usd_dir": str(usd_dir),
            "dataset_path": str(dataset_path),
            "models": ["model"],
            "event_listener": listener,
            "config": {
                "structure_assignments_path": str(assignments),
                "include_prim_path_context": False,
                "include_geometric_context": True,
                "prompts": "not-a-mapping",
                "render_mode_filter": ["composition"],
            },
        }
    )

    assert result["dataset_entries"] == []
    assert any("No image paths found" in message for message in listener.messages)


def test_validator_optional_required_field_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.config.validator as validator_mod

    monkeypatch.setattr(
        validator_mod,
        "REQUIRED_FIELDS",
        {**validator_mod.REQUIRED_FIELDS, "optional": ["field"]},
    )
    ConfigValidator().validate(
        {"project": {"name": "demo"}, "input": {"usd_path": "x"}}
    )


def test_unified_config_error_and_autowire_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = UnifiedPipelineConfigTask()
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse YAML"):
        task.run({"config_path": bad_yaml})
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration is empty"):
        task.run({"config_path": empty_yaml})

    merged = task._merge_with_defaults(
        {"project": None, "input": None, "advanced": None}
    )
    assert merged["steps"] == {}
    assert task._determine_steps(
        {"steps": {"predict": {"enabled": False}, "apply_physics": {"output": "x"}}},
        {},
    ) == ["apply_physics"]
    assert task._determine_steps(
        {"steps": {"predict": {"enabled": True}, "apply_physics": {"enabled": True}}},
        {"skip_steps": ["predict"]},
    ) == ["apply_physics"]

    resolver = SimpleNamespace(
        input_usd=tmp_path / "asset.usdz",
        get_usd_dataset_dir=lambda: tmp_path / "usd-dataset",
        get_step_output_dir=lambda _step: tmp_path / "step-out",
        get_step_dataset_file=lambda _step: tmp_path / "dataset.jsonl",
        get_predictions_dir=lambda: tmp_path / "predictions",
        get_step_predictions_file=lambda: (
            tmp_path / "predictions" / "predictions.jsonl"
        ),
        working_dir=tmp_path / "work",
        reference_images=[],
    )
    monkeypatch.setattr(
        "physics_agent.config.unified_config.RendererConfig",
        lambda **_kwargs: SimpleNamespace(
            get_rendering_modes_config=lambda _raw: {
                "composition": {},
                "linear_depth": {},
            }
        ),
    )
    build_config = task._autowire_paths(
        "build_dataset_usd",
        {"renderer": {"rendering_modes": ["composition", "linear_depth"]}},
        resolver,
        {},
    )
    assert build_config["renderer"]["rgb_rendering_modes"] == ["composition"]
    assert build_config["renderer"]["sensor_rendering_modes"] == ["linear_depth"]
    predict_config = task._autowire_paths("predict", {}, resolver, {})
    assert predict_config["output_key"] == "classification"


def test_function_inference_wrappers_delegate_with_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_classify(**kwargs: Any) -> dict[str, bool]:
        captured["classify"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(
        inference_funcs,
        "classify_object",
        fake_classify,
    )
    assert inference_funcs.classify_asset(
        vlm=object(),
        text="x",
        images=["a.png", "b.png"],
        llm=object(),
        output_key="analysis",
    ) == {"ok": True}
    assert captured["classify"]["output_key"] == "analysis"

    def fake_batch(**kwargs: Any) -> list[dict[str, str]]:
        captured["batch"] = kwargs
        return [{"status": "success"}]

    monkeypatch.setattr(inference_funcs, "batch_classify_objects", fake_batch)
    assert inference_funcs.batch_classify_assets(
        vlm=object(),
        entries=[{"id": "a"}, {"id": "b"}],
        llm=object(),
        processed_ids={"a"},
        output_key="component",
    ) == [{"status": "success"}]
    assert captured["batch"]["processed_ids"] == {"a"}


def test_scenario_resolution_usd_inspector_and_backend_edges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cap_missing_schema = BindingCapability(
        param_name="x",
        concept="x",
        binding_kind=BINDING_KIND_USD_ATTRIBUTE,
        default_range=(0.0, 1.0),
    )
    report = UsdTuningReport(
        usd_path=tmp_path / "asset.usda",
        candidates=(
            UsdTuningCandidate(
                schema="Schema",
                attribute="attr",
                prim_path="/A",
                has_authored_value=True,
                current_value=1.0,
            ),
        ),
    )
    assert report.candidates[0].to_dict()["prim"] == "/A"
    assert report.to_dict()["candidates"][0]["prim"] == "/A"
    assert (
        _resolve_usd_binding(
            capability=cap_missing_schema,
            report=report,
            backend_name="backend",
        )
        is None
    )

    sim_missing_param = BindingCapability(
        param_name="sim",
        concept="sim",
        binding_kind=BINDING_KIND_SIMULATOR_PARAMETER,
        default_range=(0.0, 1.0),
    )
    assert (
        _resolve_simulator_binding(
            capability=sim_missing_param,
            backend_name="backend",
        )
        is None
    )

    sim_cap = BindingCapability(
        param_name="sim",
        concept="sim",
        binding_kind=BINDING_KIND_SIMULATOR_PARAMETER,
        simulator_parameter="solver.x",
        default_range=(0.0, 1.0),
    )
    assert (
        _resolve_param_binding(
            param_name="sim",
            capabilities=(sim_cap,),
            report=report,
            backend_name="backend",
        )["source"]
        == "backend_capability"
    )

    bad_kind = BindingCapability(
        param_name="bad",
        concept="bad",
        binding_kind="future",
        default_range=(0.0, 1.0),
    )
    with pytest.raises(Exception, match="Could not resolve"):
        _resolve_param_binding(
            param_name="bad",
            capabilities=(bad_kind,),
            report=report,
            backend_name="backend",
        )

    caps = _backend_capabilities(
        SimpleNamespace(name="custom", tuning_capabilities=lambda: [sim_cap])
    )
    assert caps == (sim_cap,)
    with pytest.raises(Exception, match="invalid tuning capabilities"):
        _backend_capabilities(
            SimpleNamespace(name="bad", tuning_capabilities=lambda: [object()])
        )

    class BadAttr:
        def Get(self) -> None:
            raise RuntimeError("bad")

    assert _attr_value(BadAttr()) is None

    class OpinionAttr:
        def IsValid(self) -> bool:
            return True

        def HasAuthoredValueOpinion(self) -> bool:
            return True

    assert _has_authored_value(OpinionAttr()) is True

    class ExplodingAttr:
        def IsValid(self) -> bool:
            raise RuntimeError("bad")

    assert _has_authored_value(ExplodingAttr()) is False

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.endswith("newton_backend"):
            raise ImportError("no newton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(backend_mod.NewtonUnavailableError):
        backend_mod.load_newton_backend()


def test_lazy_exports_defaults_pipeline_refine_and_workflow_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import physics_agent.api as api_module
    import physics_agent.api.defaults as defaults
    import physics_agent.tasks.identify_asset as identify_mod
    import physics_agent.tasks.iterative_physics_refinement as iterative_mod
    from physics_agent.workflows.factory import (
        create_apply_physics_workflow_from_config,
    )

    pipeline_api = importlib.import_module("physics_agent.api.pipeline")
    refine_api = importlib.import_module("physics_agent.api.refine")

    assert api_module.run_tune.__module__.startswith("physics_agent")
    assert api_module.run_refine.__module__ == "physics_agent.api.refine"

    monkeypatch.setattr(defaults, "DEFAULT_VLM_BASE_URL", "", raising=False)
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY_ENV", "", raising=False)
    monkeypatch.setenv("PA_VLM_API_KEY", "direct-key")
    monkeypatch.setattr(defaults, "DEFAULT_VLM_API_KEY", "direct-key", raising=False)
    assert defaults._vlm_endpoint_config() == {"api_key_env": "${PA_VLM_API_KEY}"}
    assert defaults.os.environ["PA_VLM_API_KEY"] == "direct-key"
    assert "predict" in defaults.get_minimal_required_fields()

    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo"},
                "input": {"usd_path": "asset.usd"},
                "steps": {"predict": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )
    assert pipeline_api.PipelineInput(config=str(config_path)).config == config_path
    dry = pipeline_api._dry_run_pipeline(
        pipeline_api.PipelineInput(
            config={
                "steps": {
                    "predict": {"model": "x"},
                    "apply_physics": {"enabled": True},
                }
            },
            only_steps=["predict"],
        )
    )
    assert dry.completed_steps == ["predict"]
    assert "apply_physics" in dry.skipped_steps

    captured_contexts: list[dict[str, Any]] = []

    class Workflow:
        async def arun(self, context: dict[str, Any]) -> dict[str, Any]:
            captured_contexts.append(context)
            return {"pipeline_results": {}, "completed_steps": [], "working_dir": None}

    monkeypatch.setattr(
        "physics_agent.workflows.create_unified_pipeline_workflow",
        lambda: Workflow(),
    )
    pipeline_result = asyncio.run(
        pipeline_api.arun_pipeline(pipeline_api.PipelineInput(config=config_path))
    )
    assert pipeline_result.success is True
    assert captured_contexts[-1]["config_path"] == str(config_path)

    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        yaml.safe_dump(
            {
                "name": "drop_settle",
                "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
            }
        ),
        encoding="utf-8",
    )
    physics = tmp_path / "physics.usda"
    physics.write_text("#usda 1.0\n", encoding="utf-8")
    params = refine_api.RefineInput(
        scenario=str(scenario),
        physics_usd=physics,
        user_prompt="make it bouncy",
        output_dir=tmp_path / "refine",
        event_listener=_Listener(),
    )
    assert params.scenario == scenario

    summary = refine_api._build_iteration_summary(
        SimpleNamespace(
            iteration=1,
            iteration_dir=tmp_path,
            judge_decision="retry",
            judge_score="bad",
            judge_reasoning="r",
            best_score=float("nan"),
            n_trials=1,
            metric_name="m",
            metric_value="bad",
            cancelled=False,
            error=None,
        )
    )
    assert summary.judge_score is None
    assert summary.best_score is None
    assert summary.metric_value is None
    assert (
        refine_api._build_iteration_summary(
            SimpleNamespace(
                iteration=2,
                iteration_dir=tmp_path,
                judge_decision="retry",
                judge_score=None,
                judge_reasoning="r",
                best_score=1.0,
                n_trials=1,
                metric_name="m",
                metric_value=0.0,
                cancelled=False,
                error=None,
            )
        ).judge_score
        is None
    )

    class FakeRefineTask:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def run(self, context: dict[str, Any]) -> Any:
            assert "event_listener" in context
            return SimpleNamespace(
                iterations=[],
                termination_reason="approved",
                iteration_count=0,
                final_iteration=0,
                output_dir=tmp_path,
                final_dir=None,
                user_prompt="make it bouncy",
            )

    monkeypatch.setattr(iterative_mod, "IterativePhysicsRefinementTask", FakeRefineTask)
    assert asyncio.run(refine_api.arun_refine(params)).success is True

    monkeypatch.setattr(
        identify_mod,
        "extract_json_from_llm_response",
        lambda *_args, **_kwargs: None,
    )
    assert (
        IdentifyAssetTask()._parse_identification("raw description")[
            "asset_description"
        ]
        == "raw description"
    )
    assert [
        task.name for task in create_apply_physics_workflow_from_config().tasks
    ] == [
        "ApplyPhysicsConfig",
        "ApplyPhysics",
    ]


def test_newton_static_helper_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert NewtonSimulator._array_to_list(None, name="x") == []
    with pytest.raises(RuntimeError, match="sequence-like"):
        NewtonSimulator._array_to_list(object(), name="x")

    model = SimpleNamespace(joint_child=[0], joint_qd_start=[0])
    assert NewtonSimulator._joint_info_for_body(model, 0) == (0, 0)
    assert NewtonSimulator._joint_dof_count_for_body(model, 0) == 0
    assert (
        NewtonSimulator._eval_ik_indices_for_joint(
            SimpleNamespace(joint_articulation=[]), -1
        )
        is None
    )

    warp = ModuleType("warp")
    warp.int32 = int  # type: ignore[attr-defined]

    def array(values: list[int], **kwargs: Any) -> list[int]:
        if "device" in kwargs:
            raise TypeError("device unsupported")
        return values

    warp.array = array  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "warp", warp)
    assert NewtonSimulator._eval_ik_indices_for_joint(
        SimpleNamespace(joint_articulation=[3], device="cpu"),
        0,
    ) == [3]

    assert NewtonSimulator._ground_plane_shape_config(SimpleNamespace(), {}) is None
    assert (
        NewtonSimulator._ground_plane_shape_config(
            SimpleNamespace(),
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )

    class ShapeConfig:
        def __init__(self, **kwargs: float) -> None:
            self.kwargs = kwargs

    builder = SimpleNamespace(
        ShapeConfig=ShapeConfig,
        shape_material_ke=[],
        shape_material_kd=["bad"],
    )
    assert (
        NewtonSimulator._ground_plane_shape_config(
            builder,
            {"path_shape_map": {"/World/GroundPlane": 0}},
        )
        is None
    )

    class State:
        def __init__(self) -> None:
            self.body_qd = SimpleNamespace(
                numpy=lambda: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                assign=lambda values: setattr(self, "assigned_body_qd", values),
            )
            self.joint_q = object()
            self.joint_qd = object()

    calls: list[tuple[Any, ...]] = []
    state = State()
    NewtonSimulator._inject_initial_velocity(
        SimpleNamespace(eval_ik=lambda *args, **_kwargs: calls.append(args)),
        SimpleNamespace(joint_child=[0], joint_qd_start=[0, 6], joint_articulation=[]),
        state,
        0,
        initial_linear_velocity=(1.0, 0.0, 0.0),
        initial_angular_velocity=None,
    )
    assert calls


def test_mass_quality_usd_patch_scenario_and_backend_edges(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    import physics_agent.functions.mass_scale_quality as msq
    from physics_agent.tuning.ovphysx_backend import OvPhysXBackend
    from physics_agent.tuning.usd_patch import patch_physics_usd

    assert msq._as_float(float("inf")) is None
    assert (
        msq.extract_bbox_metrics_meters(
            {"metadata": {"world_bbox_meters": {"size": [1, 2]}}}
        )
        == {}
    )
    assert (
        msq.extract_bbox_metrics_meters(
            {"metadata": {"world_bbox_meters": {"size": [-1, 2, 3]}}}
        )
        == {}
    )
    assert (
        msq.extract_bbox_metrics_meters(
            {"text": "Dimensions (meters): width=1m, height=nanm, depth=3m"}
        )
        == {}
    )
    assert (
        msq.extract_bbox_metrics_meters(
            {"text": "Dimensions (meters): width=1em, height=2m, depth=3m"}
        )
        == {}
    )
    assert backend_mod.load_newton_backend().__class__.__name__ == "NewtonBackend"

    backend = OvPhysXBackend.__new__(OvPhysXBackend)
    calls: list[str] = []
    backend._daemon = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    backend.shutdown()
    assert calls == ["shutdown"]
    assert backend._daemon is None

    input_usd = tmp_path / "input.usda"
    stage = Usd.Stage.CreateNew(str(input_usd))
    cube = UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim()
    UsdPhysics.MassAPI.Apply(cube)
    stage.SetDefaultPrim(cube)
    stage.GetRootLayer().Save()
    output_usd = tmp_path / "output.usda"
    patch_physics_usd(input_usd, tmp_path / "mass.usda", {"mass_scale": 2.0})
    patch_physics_usd(input_usd, output_usd, {"unused": 1.0}, bindings=[])
    assert output_usd.exists()
    patch_physics_usd(input_usd, tmp_path / "skip.usda", {}, bindings=[{"param": "x"}])

    assert (
        parse_scenario(
            {
                "name": "drop_settle",
                "target": None,
                "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
            }
        ).target
        == {}
    )
    assert (
        parse_scenario(
            {
                "name": "drop_settle",
                "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
            }
        )
        .param_dict()["mass_scale"]
        .min_value
        == 0.5
    )


def _install_fake_scenario_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import world_understanding.functions.physics.trajectory as trajectory_mod

    import physics_agent.recording as recording
    import physics_agent.tuning.scenarios._scene_builder as scene_builder
    import physics_agent.tuning.usd_patch as usd_patch

    monkeypatch.setattr(
        usd_patch,
        "patch_physics_usd",
        lambda _src, dst, _params, **_kwargs: Path(dst).write_text(
            "#usda 1.0\n", encoding="utf-8"
        ),
    )
    scene_info = {
        "body_pattern": "/World/Body",
        "body_prim_path": "/World/Body",
        "rest_position": [0.0, 1.0, 0.0],
        "bbox_min_local_stage": [0.0, 0.0, 0.0],
        "bbox_max_local_stage": [1.0, 1.0, 1.0],
        "bbox_size_m": [1.0, 1.0, 1.0],
        "drop_height_m_resolved": 1.0,
        "world_up": ["bad", 0.0, 1.0],
        "camera_paths": ["/Camera"],
    }
    monkeypatch.setattr(
        scene_builder,
        "build_drop_settle_scene",
        lambda *_args, **_kwargs: dict(scene_info),
    )
    monkeypatch.setattr(
        scene_builder,
        "build_freeform_scene",
        lambda *_args, **_kwargs: dict(scene_info),
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_usda",
        lambda *_args, **_kwargs: Path(_args[3]).write_text("rec", encoding="utf-8"),
    )
    monkeypatch.setattr(
        recording,
        "author_trajectory_jsonl",
        lambda *_args, **_kwargs: Path(_args[1]).write_text("{}", encoding="utf-8"),
    )
    monkeypatch.setattr(
        trajectory_mod,
        "trajectory_summary",
        lambda _trajectory, world_up=None: {
            "final_position": [0.0, 0.0, 0.0],
            "settle_time_s": 0.5,
            "duration_s": 1.0,
            "n_samples": 2,
            "fell_over": False,
            "world_up": world_up,
        },
    )
    (tmp_path / "physics.usda").write_text("#usda 1.0\n", encoding="utf-8")


def test_drop_settle_and_freeform_render_unavailable_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_scenario_runtime(monkeypatch, tmp_path)
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "world_understanding.functions.graphics":
            raise ImportError("no graphics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    simulator = SimpleNamespace(
        evaluate=lambda **_kwargs: {
            "trajectory": [
                (0.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
                (1.0, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            ],
            "final_pose": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    drop = parse_scenario(
        {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {"vlm_check": "end_of_tune", "record_video": "end_of_tune"},
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    drop_result = drop_settle.evaluate(
        {},
        drop,
        tmp_path / "physics.usda",
        seed=1,
        simulator=simulator,
        work_dir=tmp_path / "drop",
    )
    assert drop_result["vlm_check"]["status"] == "skipped"
    assert drop_result["record_video"]["status"] == "skipped"

    free = parse_scenario(
        {
            "name": "freeform",
            "metric": "judge_score",
            "target": {
                "description": "stay upright",
                "observations": 3,
                "record_video": "always",
            },
            "parameters": [{"name": "mass_scale", "min": 0.5, "max": 2.0}],
        }
    )
    free_result = freeform.evaluate(
        {},
        free,
        tmp_path / "physics.usda",
        seed=2,
        simulator=simulator,
        work_dir=tmp_path / "free",
        judge_callback=lambda *_args: {"score": 0.1},
    )
    assert "VLM unavailable" in free_result["reasoning"]
    assert free_result["record_video"]["status"] == "skipped"


def test_visual_reporting_and_optimizer_reexport_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import world_understanding.agentic.config as agentic_config
    import world_understanding.functions.models.vision_language_models as vlm_models
    import world_understanding.utils.credentials as credentials

    import physics_agent.tasks.optimizer_models as optimizer_models
    from physics_agent.tuning.visual_evidence import (
        JudgeVisualEvidence,
        _copy_file,
        _short_label,
        prepare_reference_media,
        resolve_default_judge_vlm,
        write_comparison_contact_sheet,
    )

    assert optimizer_models.SceneOptimizerSettings is not None
    assert prepare_reference_media(output_dir=tmp_path) == JudgeVisualEvidence()
    stale_dir = tmp_path / "reference_media" / "images"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.txt").write_text("stale", encoding="utf-8")
    ref_img = tmp_path / "ref.png"
    ref_img.write_bytes(b"png")
    evidence = prepare_reference_media(output_dir=tmp_path, reference_images=[ref_img])
    assert evidence.reference_image_caption_pairs
    assert not (stale_dir / "stale.txt").exists()
    assert write_comparison_contact_sheet(
        JudgeVisualEvidence(), tmp_path / "sheet.png"
    ) == (
        None,
        None,
    )
    src = tmp_path / "same.png"
    src.write_bytes(b"png")
    _copy_file(src, src)
    assert _short_label("x" * 80).endswith("...")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        agentic_config,
        "get_api_key_for_model_config",
        lambda *_args: "api-key",
    )
    monkeypatch.setattr(
        credentials,
        "apply_vlm_nim_env_override",
        lambda cfg: {
            **cfg,
            "backend": "test-reasoning-provider",
            "base_url": "https://example.invalid",
            "reasoning_effort": "low",
            "provider_options": {"scope": "x"},
        },
    )
    monkeypatch.setattr(
        "physics_agent.tuning.visual_evidence.backend_supports_reasoning_effort",
        lambda backend: backend == "test-reasoning-provider",
    )
    monkeypatch.setattr(
        vlm_models,
        "create_vlm",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or object(),
    )
    resolve_default_judge_vlm()
    assert captured["kwargs"]["reasoning_effort"] == "low"
    assert captured["kwargs"]["provider_options"] == {"scope": "x"}

    html = GeneratePredictionReportTask()._generate_html(
        predictions=[{"id": "a", "classification": {"component_type": "x"}}],
        predictions_count=0,
        failed_count=0,
        token_stats={},
        output_key="classification",
        dataset_map={},
        dataset_path=None,
        system_prompt=None,
    )
    assert "no-data" in html
