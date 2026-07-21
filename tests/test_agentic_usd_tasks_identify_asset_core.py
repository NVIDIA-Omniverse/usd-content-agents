# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from world_understanding.agentic.usd_tasks import identify_asset as ia


class _Listener:
    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)


class _FakeVLM:
    model_name = "fake-vlm"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def generate_with_image_caption_pairs(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_identify_asset_rejects_invalid_vlm() -> None:
    with pytest.raises(TypeError, match="vlm must be"):
        ia.IdentifyAssetTask().run(
            {"vlm": object(), "composition_images": ["preview.png"]}
        )


def test_identify_asset_no_images_writes_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    monkeypatch.setattr(ia, "get_listener", lambda context, logger_name=None: listener)

    context = ia.IdentifyAssetTask().run(
        {"output_dir": tmp_path, "vlm": _FakeVLM("{}")}
    )

    output = json.loads((tmp_path / "identification.json").read_text(encoding="utf-8"))
    assert context["identification"]["asset_type"] == "unknown"
    assert context["identification_path"] == str(tmp_path / "identification.json")
    assert output["reasoning"] == "No preview images available"


def test_identify_asset_provisions_vlm_from_dict_and_parses_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    listener = _Listener()
    fake_vlm = _FakeVLM(
        json.dumps(
            {
                "asset_type": "vehicle",
                "asset_subtype": "forklift",
                "asset_description": "compact lift",
                "expected_colors": "yellow and black",
                "confidence": "high",
                "reasoning": "visible mast",
            }
        )
    )
    captured: dict[str, Any] = {}

    from world_understanding.functions.models import vision_language_models

    def fake_create_vlm(backend: str, **kwargs: Any) -> _FakeVLM:
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return fake_vlm

    monkeypatch.setattr(ia, "get_listener", lambda context, logger_name=None: listener)
    monkeypatch.setattr(ia, "get_api_key_for_model_config", lambda *args: "api-key")
    monkeypatch.setattr(vision_language_models, "create_vlm", fake_create_vlm)
    monkeypatch.setattr(
        ia.IdentifyAssetTask,
        "_extract_prim_name_hints",
        lambda self, path: [f"part{i}" for i in range(25)],
    )

    context = ia.IdentifyAssetTask().run(
        {
            "vlm": {"backend": "nim", "model": "vlm-model", "timeout": 3},
            "composition_images": [
                "preview1.png",
                "preview2.png",
                "preview3.png",
                "preview4.png",
                "preview5.png",
            ],
            "reference_images": ["ref1.png", "ref2.png", "ref3.png"],
            "usd_path": "forklift_asset.usd",
            "identify_system_prompt": "system",
            "output_dir": tmp_path,
        }
    )

    assert captured == {
        "backend": "nim",
        "kwargs": {"model": "vlm-model", "timeout": 3, "api_key": "api-key"},
    }
    assert context["identification"]["asset_subtype"] == "forklift"
    assert "forklift" in context["image_gen_prompt"]
    call = fake_vlm.calls[0]
    assert len(call["image_caption_pairs"]) == 6
    assert call["image_caption_pairs"][0] == ("Scene reference image 1:", "ref1.png")
    assert call["image_caption_pairs"][2] == (
        "3D preview of the asset (view 1):",
        "preview1.png",
    )
    assert "forklift_asset" in call["final_prompt"]
    assert "part0" in call["final_prompt"]
    assert "(25 total)" in call["final_prompt"]
    assert call["system_prompt"] == "system"


def test_identify_asset_model_get_fallback_when_config_hides_model_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WeirdConfig(dict):
        def __contains__(self, key: object) -> bool:
            if key == "model":
                return False
            return super().__contains__(key)

        def get(self, key: str, default: Any = None) -> Any:
            if key == "model":
                return "hidden-model"
            return super().get(key, default)

    captured: dict[str, Any] = {}
    fake_vlm = _FakeVLM('{"asset_type": "tool", "asset_subtype": "drill"}')

    from world_understanding.functions.models import vision_language_models

    monkeypatch.setattr(
        ia, "get_listener", lambda context, logger_name=None: _Listener()
    )
    monkeypatch.setattr(ia, "get_api_key_for_model_config", lambda *args: None)
    monkeypatch.setattr(
        ia,
        "apply_vlm_nim_env_override",
        lambda config: WeirdConfig({"backend": "custom"}),
    )
    monkeypatch.setattr(
        vision_language_models,
        "create_vlm",
        lambda backend, **kwargs: captured.update({"kwargs": kwargs}) or fake_vlm,
    )
    monkeypatch.setattr(
        ia.IdentifyAssetTask, "_extract_prim_name_hints", lambda self, path: []
    )

    ia.IdentifyAssetTask().run(
        {
            "vlm": WeirdConfig({"backend": "custom"}),
            "composition_images": ["preview.png"],
            "output_dir": tmp_path,
        }
    )

    assert captured["kwargs"]["model"] == "hidden-model"


def test_identify_asset_uses_fallback_when_vlm_call_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ia, "get_listener", lambda context, logger_name=None: _Listener()
    )
    task = ia.IdentifyAssetTask()
    monkeypatch.setattr(task, "_extract_prim_name_hints", lambda path: [])
    context = task.run(
        {
            "vlm": _FakeVLM(RuntimeError("vlm failed")),
            "rendered_preview_paths": ["preview.png"],
            "output_dir": tmp_path,
        }
    )

    assert context["identification"]["confidence"] == "low"
    assert "vlm failed" in context["identification"]["asset_description"]
    assert "the object shown" in context["image_gen_prompt"]


def test_identify_asset_parsing_and_prompt_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ia.IdentifyAssetTask()
    parsed = task._parse_identification(
        '{"asset_type": "tool", "asset_subtype": "drill"}'
    )
    assert parsed == {
        "asset_type": "tool",
        "asset_subtype": "drill",
        "asset_description": "",
        "expected_colors": "",
        "confidence": "medium",
        "reasoning": "",
    }

    monkeypatch.setattr(
        ia, "extract_json_from_llm_response", lambda *args, **kwargs: None
    )
    fallback = task._parse_identification("not json")
    assert fallback["asset_type"] == "unknown"
    assert "Could not parse response" in fallback["reasoning"]

    assert "a vehicle" in task._build_image_gen_prompt(
        {"asset_type": "vehicle", "asset_subtype": "unknown"}
    )
    assert "the object shown" in task._build_image_gen_prompt(
        {"asset_type": "unknown", "asset_subtype": "unknown"}
    )
    assert "A scene reference image" in task._build_user_prompt("asset", [], True)
    assert "HINT: The USD file" in task._build_user_prompt("asset", [], False)


def test_extract_prim_name_hints_success_missing_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = ia.IdentifyAssetTask()
    assert task._extract_prim_name_hints("") == []
    assert task._extract_prim_name_hints(str(tmp_path / "missing.usd")) == []

    usd_path = tmp_path / "scene.usd"
    usd_path.write_text("#usda", encoding="utf-8")

    class _Parent:
        def __init__(self, name: str) -> None:
            self.name = name

        def GetName(self) -> str:
            return self.name

    class _Prim:
        def __init__(
            self, name: str, *, mesh: bool, parent: _Parent | None = None
        ) -> None:
            self.name = name
            self.mesh = mesh
            self.parent = parent

        def IsA(self, schema: object) -> bool:
            return self.mesh and schema == "mesh"

        def GetParent(self) -> _Parent | None:
            return self.parent

        def GetName(self) -> str:
            return self.name

    class _Stage:
        def Traverse(self) -> list[_Prim]:
            return [
                _Prim("MeshA", mesh=True, parent=_Parent("Wheel")),
                _Prim("MeshA2", mesh=True, parent=_Parent("Wheel")),
                _Prim("LooseMesh", mesh=True),
                _Prim("Light", mesh=False),
            ]

    fake_usd = SimpleNamespace(Stage=SimpleNamespace(Open=lambda path: _Stage()))
    fake_usd_geom = SimpleNamespace(Mesh="mesh")
    monkeypatch.setattr(sys.modules["pxr"], "Usd", fake_usd, raising=False)
    monkeypatch.setattr(sys.modules["pxr"], "UsdGeom", fake_usd_geom, raising=False)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    assert task._extract_prim_name_hints(str(usd_path)) == ["Wheel", "LooseMesh"]

    fake_usd.Stage.Open = lambda path: (_ for _ in ()).throw(RuntimeError("bad usd"))
    assert task._extract_prim_name_hints(str(usd_path)) == []
