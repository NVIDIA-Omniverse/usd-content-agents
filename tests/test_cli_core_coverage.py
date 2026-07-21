# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-oriented tests for the top-level World Understanding CLI."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import typer
from PIL import Image
from pxr import Usd, UsdGeom
from pydantic import BaseModel, Field

import world_understanding.cli as cli


class _InputModel(BaseModel):
    image: dict = Field(default_factory=dict, description="image object")
    prompt: str
    backend: str = "echo"
    model: str | None = None
    target_color: list[int] = Field(default_factory=list, description="RGB color")
    colors: list[str] = Field(default_factory=list)
    count: int = Field(default=2, ge=1, le=9)
    ratio: float = 0.5
    enabled: bool = True
    names: list[str] = Field(default_factory=list)
    values: list[int] = Field(default_factory=list)
    payload: dict = Field(default_factory=dict)
    optional_text: str = "default"


class _OutputModel(BaseModel):
    answer: str = "ok"


class _ManualSchemaModel:
    @staticmethod
    def model_json_schema() -> dict[str, object]:
        return {
            "required": ["required_text"],
            "properties": {
                "image": {"type": "object", "properties": {"path": {"type": "string"}}},
                "target_color": {"type": "string", "description": "CSS swatch"},
                "bounded": {"type": "integer", "min": 2, "max": 8},
                "misc": {"type": "array", "items": {"type": "object"}},
                "required_text": {"type": "string"},
            },
        }


class _FakeResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    def model_dump(self) -> dict[str, object]:
        return dict(self.__dict__)


class _FakeTool:
    def __init__(
        self, result: object | None = None, *, raises: Exception | None = None
    ):
        self.spec = SimpleNamespace(
            name="fake",
            version="1.0",
            description="Fake tool",
            tags=["unit"],
            input_model=_InputModel,
            output_model=_OutputModel,
        )
        self._result = result or _FakeResult(response="tool response")
        self._raises = raises
        self.calls: list[dict[str, object]] = []

    def to_json_schema(self) -> dict[str, object]:
        return {"type": "object", "title": "FakeTool"}

    def run(self, inputs: dict[str, object]) -> object:
        self.calls.append(inputs)
        if self._raises:
            raise self._raises
        return self._result


class _FakeRegistry:
    def __init__(self, tools: dict[str, object]):
        self._tools = tools

    def list_tools(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> object | None:
        return self._tools.get(name)


class _FakeDisplayRegistry:
    def __init__(self, value: bool = False):
        self.value = value
        self.calls: list[tuple[str, object]] = []

    def display(self, name: str, result: object, console: object) -> bool:
        self.calls.append((name, result))
        return self.value


def _patch_console(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    printed: list[str] = []
    monkeypatch.setattr(
        cli.console,
        "print",
        lambda *args, **kwargs: printed.append(" ".join(map(str, args))),
    )
    monkeypatch.setattr(
        cli.console,
        "print_json",
        lambda *args, **kwargs: printed.append("json"),
    )
    monkeypatch.setattr(cli.console, "print_exception", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "print", lambda *args, **kwargs: printed.append("rich"))
    return printed


def _write_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Mesh.Define(stage, "/World/Mesh")
    UsdGeom.Camera.Define(stage, "/World/Camera")
    stage.GetRootLayer().Save()
    return path


def test_cli_helpers_and_tool_registry_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_console(monkeypatch)
    assert cli._backend_help("public") == "public; installed plugins may add providers"

    assert cli._format_file_size(42) == "42 B"
    assert cli._format_file_size(2048) == "2.0 KB"
    assert cli._format_file_size(2 * 1024 * 1024) == "2.0 MB"
    assert cli._generate_example_input(None) == {}
    example = cli._generate_example_input(_InputModel)
    assert example["prompt"] == "Your prompt text here"
    assert example["target_color"] == [255, 87, 51]
    assert example["values"] == [1, 2, 3]

    with pytest.raises(typer.Exit):
        cli.version_callback(True)
    with pytest.raises(typer.Exit):
        cli.main(log_level="NOPE")

    good_tool = _FakeTool()
    registry = _FakeRegistry({"good": good_tool, "missing-spec": object()})
    monkeypatch.setattr(cli, "setup_registry", lambda: registry)
    cli.list_tools(verbose=True)

    with pytest.raises(typer.Exit):
        cli.tool_info("absent")
    cli.tool_info("good", show_schema=True, example_input=True, verbose=True)


def test_cli_schema_and_registry_setup_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_console(monkeypatch)
    example = cli._generate_example_input(_ManualSchemaModel)
    assert example["image"] == {
        "path": "path/to/image.jpg",
        "width": 1920,
        "height": 1080,
    }
    assert example["target_color"] == "#FF5733"
    assert example["bounded"] == 5
    assert example["misc"] == []
    assert example["required_text"] == "example_required_text"

    registry = object()
    monkeypatch.setattr(cli, "get_tool_registry", lambda: registry)
    monkeypatch.setattr(cli, "register_all_tools", lambda: ["one", "two"])
    assert cli.setup_registry() is registry


class _FakeRootLayer:
    identifier = "fake-layer.usda"

    def __init__(self, *, export_result: bool = True, create_output: bool = True):
        self.export_result = export_result
        self.create_output = create_output

    def Export(self, path: str) -> bool:
        if self.create_output:
            Path(path).write_text("#usda 1.0\n", encoding="utf-8")
        return self.export_result


class _FakeStage:
    def __init__(self, root_layer: _FakeRootLayer):
        self._root_layer = root_layer

    def GetRootLayer(self) -> _FakeRootLayer:
        return self._root_layer

    def GetTimeCodesPerSecond(self) -> float:
        return 24.0

    def GetFramesPerSecond(self) -> float:
        return 24.0

    def GetStartTimeCode(self) -> float:
        return 0.0

    def GetEndTimeCode(self) -> float:
        return 1.0

    def Traverse(self) -> list[object]:
        return [object(), object()]

    def Flatten(self) -> _FakeRootLayer:
        return self._root_layer


def test_cli_convert_usd_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    source = _write_stage(tmp_path / "scene.usda")
    destination = tmp_path / "nested" / "scene.usdc"
    cli.convert_usd(str(source), str(destination), force=True, verbose=True)
    assert destination.exists()

    with pytest.raises(typer.Exit):
        cli.convert_usd(
            str(tmp_path / "missing.usda"),
            str(tmp_path / "out.usda"),
            force=False,
            verbose=False,
        )

    existing = tmp_path / "existing.usda"
    existing.write_text("already here", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli.convert_usd(str(source), str(existing), force=False, verbose=False)

    bad_source = tmp_path / "bad.txt"
    bad_source.write_text("not usd", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli.convert_usd(str(bad_source), str(tmp_path / "out.usda"), False, False)

    with pytest.raises(typer.Exit):
        cli.convert_usd(str(source), str(tmp_path / "out.txt"), False, False)

    from pxr import Usd as PxrUsd

    monkeypatch.setattr(PxrUsd.Stage, "Open", lambda *args, **kwargs: None)
    with pytest.raises(typer.Exit):
        cli.convert_usd(str(source), str(tmp_path / "open_failed.usda"), True, False)

    monkeypatch.setattr(
        PxrUsd.Stage,
        "Open",
        lambda *args, **kwargs: _FakeStage(
            _FakeRootLayer(export_result=False, create_output=False)
        ),
    )
    with pytest.raises(typer.Exit):
        cli.convert_usd(str(source), str(tmp_path / "export_failed.usda"), True, False)

    monkeypatch.setattr(
        PxrUsd.Stage,
        "Open",
        lambda *args, **kwargs: _FakeStage(
            _FakeRootLayer(export_result=True, create_output=False)
        ),
    )
    with pytest.raises(typer.Exit):
        cli.convert_usd(str(source), str(tmp_path / "not_created.usda"), True, False)


def test_cli_convert_usd_import_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pxr" and "Usd" in fromlist:
            raise ImportError("no usd")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(typer.Exit):
        cli.convert_usd(
            str(tmp_path / "scene.usda"),
            str(tmp_path / "scene.usdc"),
            force=False,
            verbose=False,
        )


def test_cli_flatten_usd_extra_error_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    source = _write_stage(tmp_path / "scene.usda")

    with pytest.raises(typer.Exit):
        cli.flatten_usd(
            str(source), str(tmp_path / "flat.txt"), force=False, verbose=False
        )

    from pxr import Usd as PxrUsd

    monkeypatch.setattr(PxrUsd.Stage, "Open", lambda *args, **kwargs: None)
    with pytest.raises(typer.Exit):
        cli.flatten_usd(str(source), str(tmp_path / "open_failed.usda"), True, False)

    monkeypatch.setattr(
        PxrUsd.Stage,
        "Open",
        lambda *args, **kwargs: _FakeStage(
            _FakeRootLayer(export_result=False, create_output=False)
        ),
    )
    with pytest.raises(typer.Exit):
        cli.flatten_usd(str(source), str(tmp_path / "export_failed.usda"), True, False)

    monkeypatch.setattr(
        PxrUsd.Stage,
        "Open",
        lambda *args, **kwargs: _FakeStage(
            _FakeRootLayer(export_result=True, create_output=False)
        ),
    )
    with pytest.raises(typer.Exit):
        cli.flatten_usd(str(source), str(tmp_path / "not_created.usda"), True, False)


def test_cli_run_tool_chat_vision_detect_and_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    display = _FakeDisplayRegistry(False)
    monkeypatch.setattr(cli, "get_display_registry", lambda: display)

    inputs_file = tmp_path / "inputs.json"
    inputs_file.write_text(json.dumps({"x": 1}), encoding="utf-8")
    tool = _FakeTool(_FakeResult(response="ok", value=1))
    registry = _FakeRegistry({"tool": tool})
    monkeypatch.setattr(cli, "setup_registry", lambda: registry)
    cli.run_tool("tool", inputs_file=inputs_file, output_format="json", verbose=True)
    cli.run_tool("tool", inputs_file=None, output_format="text", verbose=False)
    assert tool.calls[-1] == {}
    with pytest.raises(typer.Exit):
        cli.run_tool("missing", inputs_file=None, output_format="text", verbose=False)
    monkeypatch.setattr(
        cli,
        "setup_registry",
        lambda: _FakeRegistry({"boom": _FakeTool(raises=RuntimeError("boom"))}),
    )
    with pytest.raises(typer.Exit):
        cli.run_tool("boom", inputs_file=None, output_format="text", verbose=False)

    chat_tool = _FakeTool({"response": "dict response"})
    monkeypatch.setattr(
        cli, "setup_registry", lambda: _FakeRegistry({"chat": chat_tool})
    )
    cli.chat("hello", backend="echo", model="m", verbose=True)
    chat_tool._result = SimpleNamespace(response="object response")
    cli.chat(
        "hello",
        backend="nim",
        model=None,
        temperature=0.7,
        max_tokens=500,
        verbose=False,
    )
    chat_tool._result = "plain"
    cli.chat(
        "hello",
        backend="nim",
        model=None,
        temperature=0.7,
        max_tokens=500,
        verbose=False,
    )
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({}))
    with pytest.raises(typer.Exit):
        cli.chat(
            "hello",
            backend="nim",
            model=None,
            temperature=0.7,
            max_tokens=500,
            verbose=False,
        )

    vlm_result = _FakeResult(response="vision response")
    vlm_tool = _FakeTool(vlm_result)
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({"vlm": vlm_tool}))
    cli.vision(
        "image.png",
        prompt="Describe this image in detail.",
        backend="nim",
        model="vlm-model",
        system_prompt="You are a helpful AI assistant that can analyze images.",
        temperature=0.7,
        max_tokens=1024,
        output_format="json",
        verbose=True,
    )
    cli.vision(
        "image.png",
        prompt="Describe this image in detail.",
        backend="nim",
        model=None,
        system_prompt="You are a helpful AI assistant that can analyze images.",
        temperature=0.7,
        max_tokens=1024,
        output_format="text",
        verbose=False,
    )
    with pytest.raises(typer.Exit):
        vlm_tool._raises = RuntimeError("vision bad")
        cli.vision(
            "image.png",
            prompt="Describe this image in detail.",
            backend="nim",
            model=None,
            system_prompt="You are a helpful AI assistant that can analyze images.",
            temperature=0.7,
            max_tokens=1024,
            output_format="text",
            verbose=False,
        )
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({}))
    with pytest.raises(typer.Exit):
        cli.vision(
            "image.png",
            prompt="Describe this image in detail.",
            backend="nim",
            model=None,
            system_prompt="You are a helpful AI assistant that can analyze images.",
            temperature=0.7,
            max_tokens=1024,
            output_format="text",
            verbose=False,
        )

    dino_tool = _FakeTool(_FakeResult(boxes=[]))
    monkeypatch.setattr(
        cli, "setup_registry", lambda: _FakeRegistry({"grounding_dino": dino_tool})
    )
    cli.detect("image.png", "chair", threshold=0.3, output_format="json", verbose=True)
    cli.detect("image.png", "chair", threshold=0.3, output_format="text", verbose=False)
    with pytest.raises(typer.Exit):
        dino_tool._raises = RuntimeError("detect bad")
        cli.detect(
            "image.png", "chair", threshold=0.3, output_format="text", verbose=False
        )
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({}))
    with pytest.raises(typer.Exit):
        cli.detect(
            "image.png", "chair", threshold=0.3, output_format="text", verbose=False
        )

    search_result = SimpleNamespace(
        success=True,
        errors=[],
        num_results=3,
        processing_time_ms=12.3,
        file_extensions=["usd", "mdl"],
        results=[
            {
                "source": {
                    "name": "Tiny",
                    "path": "/tiny.usd",
                    "ext": "usd",
                    "size": 100,
                    "modified_timestamp": "now",
                },
                "score": 0.9,
                "metadata": {"rrf_score": 0.123456, "rrf_rank": 1},
                "thumbnail_exists": True,
                "id": "tiny",
            },
            {
                "source": {
                    "name": "Mid",
                    "path": "/mid.usd",
                    "ext": "usd",
                    "size": 2048,
                },
                "thumbnail_exists": False,
            },
            {
                "source": {
                    "name": "Big",
                    "path": "/big.usd",
                    "ext": "usd",
                    "size": 3 * 1024 * 1024,
                },
            },
        ],
        model_dump_json=lambda indent=None: "{}",
    )
    search_tool = _FakeTool(search_result)
    monkeypatch.setattr(
        cli, "setup_registry", lambda: _FakeRegistry({"usd_search": search_tool})
    )
    cli.usd_search(
        "chair",
        limit=10,
        api_host="https://example",
        file_extensions="usd, mdl",
        no_metadata=False,
        no_images=False,
        output_format="text",
        verbose=True,
    )
    cli.usd_search(
        "chair",
        limit=10,
        api_host=None,
        file_extensions=None,
        no_metadata=False,
        no_images=False,
        output_format="json",
        verbose=False,
    )
    search_tool._result = SimpleNamespace(success=True, num_results=0, results=[])
    cli.usd_search(
        "none",
        limit=10,
        api_host=None,
        file_extensions=None,
        no_metadata=False,
        no_images=False,
        output_format="text",
        verbose=False,
    )
    search_tool._result = SimpleNamespace(success=False, errors=["bad"])
    with pytest.raises(typer.Exit):
        cli.usd_search(
            "bad",
            limit=10,
            api_host=None,
            file_extensions=None,
            no_metadata=False,
            no_images=False,
            output_format="text",
            verbose=False,
        )
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({}))
    with pytest.raises(typer.Exit):
        cli.usd_search(
            "missing",
            limit=10,
            api_host=None,
            file_extensions=None,
            no_metadata=False,
            no_images=False,
            output_format="text",
            verbose=False,
        )


def test_cli_nat_and_image_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    config = tmp_path / "nat.yaml"
    config.write_text("workflow: demo\n", encoding="utf-8")

    runtime_loader = ModuleType("world_understanding.nat.runtime_loader")
    runtime_loader.validate_nat_config = lambda path: True

    async def fake_query(path: Path, question: str) -> str:
        return f"answer:{question}"

    runtime_loader.query_workflow = fake_query

    class _Workflow:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def query(self, question: str) -> str:
            return f"interactive:{question}"

    runtime_loader.NATWorkflow = lambda path: _Workflow()
    monkeypatch.setitem(
        sys.modules, "world_understanding.nat.runtime_loader", runtime_loader
    )
    cli.run_nat(config, "question", validate_only=True, interactive=False, verbose=True)
    runtime_loader.validate_nat_config = lambda path: False
    with pytest.raises(typer.Exit):
        cli.run_nat(
            config, "question", validate_only=True, interactive=False, verbose=False
        )
    runtime_loader.validate_nat_config = lambda path: True
    cli.run_nat(
        config, "question", validate_only=False, interactive=False, verbose=False
    )
    prompts = iter(["follow-up", "exit"])
    monkeypatch.setattr(cli.typer, "prompt", lambda prompt: next(prompts))
    cli.run_nat(
        config, "question", validate_only=False, interactive=True, verbose=False
    )
    monkeypatch.setattr(
        cli.typer,
        "prompt",
        lambda prompt: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    cli.run_nat(
        config, "question", validate_only=False, interactive=True, verbose=False
    )
    with pytest.raises(typer.Exit):
        cli.run_nat(
            tmp_path / "missing.yaml",
            "question",
            validate_only=False,
            interactive=False,
            verbose=False,
        )

    edited = tmp_path / "edited.png"
    rescaled = tmp_path / "rescaled.png"
    Image.new("RGB", (1, 1), "red").save(edited)
    Image.new("RGB", (1, 1), "blue").save(rescaled)
    result = _FakeResult(
        edited_image_path=str(edited),
        rescaled_input_path=str(rescaled),
        execution_time=1.25,
    )
    tool = _FakeTool(result)
    monkeypatch.setattr(
        cli, "setup_registry", lambda: _FakeRegistry({"image_edit": tool})
    )
    monkeypatch.setattr(
        cli, "get_display_registry", lambda: _FakeDisplayRegistry(False)
    )
    output = tmp_path / "custom.png"
    saved_input = tmp_path / "input.png"
    cli.edit_image(
        "source.png",
        "make blue",
        output=str(output),
        save_rescaled_input=str(saved_input),
        negative_prompt="",
        server_url="http://comfy",
        verbose=True,
    )
    assert output.exists()
    assert saved_input.exists()
    monkeypatch.setattr(cli, "setup_registry", lambda: _FakeRegistry({}))
    with pytest.raises(typer.Exit):
        cli.edit_image(
            "source.png",
            "prompt",
            output=None,
            save_rescaled_input=None,
            negative_prompt="",
            server_url=None,
            verbose=False,
        )
    monkeypatch.setattr(
        cli,
        "setup_registry",
        lambda: _FakeRegistry(
            {"image_edit": _FakeTool(raises=RuntimeError("edit bad"))}
        ),
    )
    with pytest.raises(typer.Exit):
        cli.edit_image(
            "source.png",
            "prompt",
            output=None,
            save_rescaled_input=None,
            negative_prompt="",
            server_url=None,
            verbose=False,
        )


def test_cli_print_usd_query_and_tree_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    usd_path = _write_stage(tmp_path / "printable.usda")

    class _FakePrim:
        path = "/World/Mesh"
        name = "Mesh"
        type_name = "Mesh"
        is_active = True
        is_xform = False
        is_instance = False

        def get_depth(self) -> int:
            return 2

    class _FakeUSDModel:
        def __init__(self, usd_file: Path) -> None:
            self.usd_file = usd_file

        def get_prim(self, path: str) -> object | None:
            return _FakePrim() if path == "/World/Mesh" else None

        def get_parent(self, path: str) -> object:
            return SimpleNamespace(path="/World")

        def get_children(self, path: str) -> list[object]:
            return [SimpleNamespace(path=f"/World/Mesh/Child{i}") for i in range(12)]

        def get_collections_containing_prim(self, path: str) -> list[object]:
            return [
                SimpleNamespace(
                    name="owned", prim_path="/World", includes=[], excludes=[]
                ),
                SimpleNamespace(
                    name="defined", prim_path="/World/Mesh", includes=[], excludes=[]
                ),
            ]

        def get_xform_owning_collection(self, collection: object) -> object | None:
            if collection.name == "owned":
                return SimpleNamespace(path="/World")
            return None

        def get_collections_on_prim(self, path: str) -> list[object]:
            return [
                SimpleNamespace(
                    name="local",
                    includes=["/World/Mesh"],
                    excludes=["/World/Mesh/Hidden"],
                )
            ]

        def get_path_to_root(self, path: str) -> list[str]:
            return ["/", "/World", path]

        def get_subtree_stats(self, path: str) -> dict[str, int]:
            return {
                "total_prims": 13,
                "max_depth": 3,
                "num_xforms": 1,
                "num_instances": 2,
                "num_inactive": 1,
            }

        def print_tree(self, **kwargs: object) -> None:
            self.print_tree_kwargs = kwargs

        def print_summary(self) -> None:
            self.summary_printed = True

    import world_understanding.functions.graphics.usd_model as usd_model

    monkeypatch.setattr(usd_model, "USDModel", _FakeUSDModel)
    cli.print_usd(
        str(usd_path),
        start_prim=None,
        show_types=False,
        show_variants=False,
        show_api_schemas=False,
        show_collections=False,
        show_custom_tokens=False,
        show_all=False,
        active_only=False,
        max_depth=None,
        no_info=False,
        stats=False,
        query_prim="/World/Mesh",
        verbose=True,
    )
    cli.print_usd(
        str(usd_path),
        start_prim="/World",
        show_types=False,
        show_variants=False,
        show_api_schemas=False,
        show_collections=False,
        show_custom_tokens=False,
        show_all=True,
        active_only=True,
        max_depth=3,
        no_info=True,
        stats=False,
        query_prim=None,
        verbose=True,
    )
    with pytest.raises(typer.Exit):
        cli.print_usd(
            str(usd_path),
            start_prim=None,
            show_types=False,
            show_variants=False,
            show_api_schemas=False,
            show_collections=False,
            show_custom_tokens=False,
            show_all=False,
            active_only=False,
            max_depth=None,
            no_info=False,
            stats=False,
            query_prim="/Missing",
            verbose=True,
        )
    with pytest.raises(typer.Exit):
        cli.print_usd(
            str(tmp_path / "missing.usda"),
            start_prim=None,
            show_types=False,
            show_variants=False,
            show_api_schemas=False,
            show_collections=False,
            show_custom_tokens=False,
            show_all=False,
            active_only=False,
            max_depth=None,
            no_info=False,
            stats=False,
            query_prim=None,
            verbose=False,
        )


def test_cli_usd_query_inspect_summary_and_optimize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_console(monkeypatch)
    usd_path = _write_stage(tmp_path / "scene.usda")

    import world_understanding.functions.graphics.usd_spatial as usd_spatial

    query_calls: list[dict[str, object]] = []

    def fake_query_prims(stage, **kwargs):
        query_calls.append(kwargs)
        return [
            {
                "path": "/World/Mesh",
                "type": "Mesh",
                "volume": 1.2345,
                "distance": 2.0,
                "material": "/Materials/Mat",
            }
        ]

    monkeypatch.setattr(usd_spatial, "query_prims", fake_query_prims)
    cli.query_usd(
        str(usd_path),
        name="Mesh*",
        path_pattern="/World/*",
        prim_type="Mesh",
        has_material=True,
        no_material=False,
        min_size=None,
        max_size=None,
        near="1,2,3",
        radius=5,
        overlaps="/World/Other",
        sort="name",
        limit=None,
        start_prim=None,
        active_only=False,
        output_format="json",
        verbose=True,
    )
    assert query_calls[-1]["near"] == [1.0, 2.0, 3.0]
    cli.query_usd(
        str(usd_path),
        name=None,
        path_pattern=None,
        prim_type=None,
        has_material=False,
        no_material=True,
        min_size=None,
        max_size=None,
        near="/World/Mesh",
        radius=None,
        overlaps=None,
        sort="name",
        limit=None,
        start_prim=None,
        active_only=False,
        output_format="paths",
        verbose=False,
    )
    cli.query_usd(
        str(usd_path),
        name=None,
        path_pattern=None,
        prim_type=None,
        has_material=False,
        no_material=False,
        min_size=None,
        max_size=None,
        near=None,
        radius=None,
        overlaps=None,
        sort="name",
        limit=None,
        start_prim=None,
        active_only=False,
        output_format="table",
        verbose=False,
    )
    with pytest.raises(typer.Exit):
        cli.query_usd(
            str(tmp_path / "missing.usda"),
            name=None,
            path_pattern=None,
            prim_type=None,
            has_material=False,
            no_material=False,
            min_size=None,
            max_size=None,
            near=None,
            radius=None,
            overlaps=None,
            sort="name",
            limit=None,
            start_prim=None,
            active_only=False,
            output_format="json",
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.query_usd(
            str(usd_path),
            name=None,
            path_pattern=None,
            prim_type=None,
            has_material=False,
            no_material=False,
            min_size=None,
            max_size=None,
            near="1,2",
            radius=None,
            overlaps=None,
            sort="name",
            limit=None,
            start_prim=None,
            active_only=False,
            output_format="json",
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.query_usd(
            str(usd_path),
            name=None,
            path_pattern=None,
            prim_type=None,
            has_material=False,
            no_material=False,
            min_size=None,
            max_size=None,
            near="not-a-point",
            radius=None,
            overlaps=None,
            sort="name",
            limit=None,
            start_prim=None,
            active_only=False,
            output_format="json",
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.query_usd(
            str(usd_path),
            name=None,
            path_pattern=None,
            prim_type=None,
            has_material=False,
            no_material=False,
            min_size=None,
            max_size=None,
            near=None,
            radius=None,
            overlaps=None,
            sort="name",
            limit=None,
            start_prim=None,
            active_only=False,
            output_format="bad",
            verbose=False,
        )

    with monkeypatch.context() as m:
        m.setattr(Usd.Stage, "Open", lambda *args, **kwargs: None)
        with pytest.raises(typer.Exit):
            cli.query_usd(
                str(usd_path),
                name=None,
                path_pattern=None,
                prim_type=None,
                has_material=False,
                no_material=False,
                min_size=None,
                max_size=None,
                near=None,
                radius=None,
                overlaps=None,
                sort="name",
                limit=None,
                start_prim=None,
                active_only=False,
                output_format="json",
                verbose=False,
            )

    inspect_result = {
        "path": "/World/Mesh",
        "type": "Mesh",
        "active": True,
        "child_count": 0,
        "volume": 1.0,
        "material": "/Materials/Mat",
        "geometry": {"points": 3},
        "world_transform": {"translation": [0, 0, 0]},
        "properties": {"purpose": "default"},
    }
    monkeypatch.setattr(
        usd_spatial,
        "inspect_prim",
        lambda stage, path, **kwargs: inspect_result if path != "/Missing" else None,
    )
    cli.inspect_usd(
        str(usd_path),
        ["/World/Mesh"],
        world_transform=False,
        geometry=False,
        properties=False,
        show_all=True,
        output_format="json",
        verbose=False,
    )
    cli.inspect_usd(
        str(usd_path),
        ["/World/Mesh", "/Missing"],
        geometry=True,
        world_transform=True,
        properties=True,
        show_all=False,
        output_format="table",
        verbose=True,
    )
    with pytest.raises(typer.Exit):
        cli.inspect_usd(
            str(usd_path),
            ["/Missing"],
            world_transform=False,
            geometry=False,
            properties=False,
            show_all=False,
            output_format="json",
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.inspect_usd(
            str(tmp_path / "missing.usda"),
            ["/World"],
            world_transform=False,
            geometry=False,
            properties=False,
            show_all=False,
            output_format="json",
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.inspect_usd(
            str(usd_path),
            ["/World/Mesh"],
            world_transform=False,
            geometry=False,
            properties=False,
            show_all=False,
            output_format="bad",
            verbose=False,
        )

    with monkeypatch.context() as m:
        m.setattr(Usd.Stage, "Open", lambda *args, **kwargs: None)
        with pytest.raises(typer.Exit):
            cli.inspect_usd(
                str(usd_path),
                ["/World/Mesh"],
                world_transform=False,
                geometry=False,
                properties=False,
                show_all=False,
                output_format="json",
                verbose=False,
            )

    summary = {
        "stage_info": {
            "root_layer": "scene.usda",
            "up_axis": "Y",
            "meters_per_unit": 1.0,
            "start_time": 0,
            "end_time": 24,
            "fps": 24,
        },
        "composition": {
            "total_prims": 2,
            "type_counts": {"Xform": 1, "Mesh": 1},
            "instance_count": 1,
        },
        "spatial_extents": {"min": [0, 0, 0], "max": [1, 2, 3], "size": [1, 2, 3]},
        "largest_prims": [{"path": "/World/Mesh", "volume": 2.5}],
        "materials": [{"material": "/Materials/Mat", "bound_prim_count": 1}],
    }
    monkeypatch.setattr(usd_spatial, "scene_summary", lambda stage, **kwargs: summary)
    cli.scene_summary_cmd(
        str(usd_path),
        output_format="text",
        start_prim="/World",
        top_n=5,
        verbose=True,
    )
    cli.scene_summary_cmd(
        str(usd_path),
        output_format="json",
        start_prim=None,
        top_n=5,
        verbose=False,
    )
    with pytest.raises(typer.Exit):
        cli.scene_summary_cmd(
            str(tmp_path / "missing.usda"),
            output_format="text",
            start_prim=None,
            top_n=5,
            verbose=False,
        )
    with pytest.raises(typer.Exit):
        cli.scene_summary_cmd(
            str(usd_path),
            output_format="bad",
            start_prim=None,
            top_n=5,
            verbose=False,
        )

    with monkeypatch.context() as m:
        m.setattr(Usd.Stage, "Open", lambda *args, **kwargs: None)
        with pytest.raises(typer.Exit):
            cli.scene_summary_cmd(
                str(usd_path),
                output_format="text",
                start_prim=None,
                top_n=5,
                verbose=False,
            )

    import world_understanding.functions.optimization as optimization

    result = {
        "best_value": -2.0,
        "n_evals": 3,
        "elapsed": 0.1,
        "best_x": [1.234],
        "generations": 2,
    }

    def fake_optimizer(evaluate_fn, *args, **kwargs):
        evaluate_fn()
        return result

    monkeypatch.setattr(optimization, "cma_es", fake_optimizer)
    monkeypatch.setattr(optimization, "simulated_annealing", fake_optimizer)
    monkeypatch.setattr(optimization, "random_search", fake_optimizer)
    goal_file = tmp_path / "goal.py"
    goal_file.write_text(
        """
METRIC_NAME = "score"
METRIC_DIRECTION = "maximize"
TIME_BUDGET = 1.0
N_DIMS = 1
BOUNDS = (0.0, 1.0)
def evaluate(**ctx):
    return 2.0
""".strip(),
        encoding="utf-8",
    )
    cli.optimize(
        goal_file,
        algorithm="cma-es",
        time_budget=0.5,
        seed=42,
        temp_init=5.0,
        temp_final=1e-4,
        step_size=0.5,
        sigma_init=2.0,
    )
    cli.optimize(
        goal_file,
        algorithm="simulated-annealing",
        time_budget=None,
        seed=42,
        temp_init=5.0,
        temp_final=1e-4,
        step_size=0.5,
        sigma_init=2.0,
    )
    cli.optimize(
        goal_file,
        algorithm="random-search",
        time_budget=None,
        seed=42,
        temp_init=5.0,
        temp_final=1e-4,
        step_size=0.5,
        sigma_init=2.0,
    )
    class_goal = tmp_path / "class_goal.py"
    class_goal.write_text(
        """
from typing import Any
from world_understanding.functions.optimization import Goal

class TinyGoal(Goal):
    @property
    def metric_name(self) -> str:
        return "loss"

    @property
    def metric_direction(self) -> str:
        return "minimize"

    @property
    def time_budget(self) -> float:
        return 1.0

    @property
    def n_dims(self) -> int:
        return 1

    @property
    def bounds(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def evaluate(self, **context: Any) -> float:
        return 0.25
""".strip(),
        encoding="utf-8",
    )
    cli.optimize(
        class_goal,
        algorithm="random-search",
        time_budget=None,
        seed=42,
        temp_init=5.0,
        temp_final=1e-4,
        step_size=0.5,
        sigma_init=2.0,
    )
    with monkeypatch.context() as m:
        m.setattr("importlib.util.spec_from_file_location", lambda *args: None)
        with pytest.raises(typer.Exit):
            cli.optimize(
                goal_file,
                algorithm="cma-es",
                time_budget=None,
                seed=42,
                temp_init=5.0,
                temp_final=1e-4,
                step_size=0.5,
                sigma_init=2.0,
            )
    with pytest.raises(typer.Exit):
        cli.optimize(
            tmp_path / "missing_goal.py",
            algorithm="cma-es",
            time_budget=None,
            seed=42,
            temp_init=5.0,
            temp_final=1e-4,
            step_size=0.5,
            sigma_init=2.0,
        )
    bad_goal = tmp_path / "bad_goal.py"
    bad_goal.write_text("METRIC_NAME = 'score'\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli.optimize(
            bad_goal,
            algorithm="cma-es",
            time_budget=None,
            seed=42,
            temp_init=5.0,
            temp_final=1e-4,
            step_size=0.5,
            sigma_init=2.0,
        )
    with pytest.raises(typer.Exit):
        cli.optimize(
            goal_file,
            algorithm="unknown",
            time_budget=None,
            seed=42,
            temp_init=5.0,
            temp_final=1e-4,
            step_size=0.5,
            sigma_init=2.0,
        )
