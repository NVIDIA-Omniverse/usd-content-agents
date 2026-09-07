# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from texture_agent.config.rendering_backends import validate_texture_rendering_steps
from texture_agent.functions import artifact_manifest
from texture_agent.functions.cached_apply import is_valid_cached_texture_png
from texture_agent.tasks import apply_textures, generate_prompts, generate_textures


def test_rendering_step_validation_ignores_legacy_non_mapping_step() -> None:
    validate_texture_rendering_steps({"render": "legacy-disabled-step"})


def test_partial_artifact_discovery_handles_directory_scan_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("scan failed")),
    )

    assert (
        artifact_manifest._discover_partial_artifacts({"working_dir": str(tmp_path)})
        == []
    )


def test_partial_artifact_discovery_skips_manifests_and_unreadable_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "artifacts_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / ".artifacts_manifest.json.partial").write_text("{}", encoding="utf-8")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"bad")
    good = tmp_path / "good.bin"
    good.write_bytes(b"good")
    real_resolve = Path.resolve

    def _resolve(path: Path, *args: object, **kwargs: object) -> Path:
        if path == bad:
            raise OSError("unreadable")
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _resolve)

    assert artifact_manifest._discover_partial_artifacts(
        {"working_dir": str(tmp_path)}
    ) == [{"path": "good.bin", "size_bytes": 4}]


def test_atomic_manifest_cleanup_tolerates_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "manifest.json"
    real_unlink = Path.unlink

    def _unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".manifest.json."):
            raise OSError("cleanup failed")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", _unlink)

    artifact_manifest._write_json_atomically(target, {"ok": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_cached_png_rejects_image_above_pixel_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (2, 2)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3)

    assert is_valid_cached_texture_png(path) is False


def test_preview_graph_tolerates_channel_removed_after_selection(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdShade

    class _ChangingPaths(dict[str, str]):
        def __init__(self) -> None:
            super().__init__({"albedo": "albedo.png"})
            self._reads = 0

        def get(self, key: str, default: str | None = None) -> str | None:
            if key == "albedo":
                self._reads += 1
                return "albedo.png" if self._reads == 1 else None
            return super().get(key, default)

    stage = Usd.Stage.CreateNew(str(tmp_path / "preview.usda"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    preview = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")

    assert (
        apply_textures._author_usd_preview_texture_graph(
            stage,
            str(material.GetPath()),
            [(preview, frozenset({"albedo"}))],
            _ChangingPaths(),
        )
        == []
    )


@pytest.mark.parametrize(
    ("context", "working_dir"),
    [
        (
            {
                "planning_config": {
                    "resume_apply_textures": True,
                    "apply_texture_plan_unit_ids": False,
                }
            },
            None,
        ),
        ({"resume": True, "texture_plan_path": "missing-plan.json"}, None),
        ({"resume": True}, None),
    ],
)
def test_resumed_plan_loader_handles_legacy_and_missing_paths(
    context: dict[str, object],
    working_dir: str | Path | None,
) -> None:
    assert (
        generate_prompts._load_resumed_texture_plan(
            context,
            working_dir=working_dir,
        )
        is None
    )


def test_simple_image_gen_capabilities_reject_non_mapping_provider_config() -> None:
    assert (
        generate_textures._simple_image_gen_conditioning_capabilities(
            {"backend": "simple_image_gen", "image_gen": ["legacy"]}
        )
        is None
    )
