# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from content_workflow_cli import vision_observer


def _valid_observations() -> dict[str, object]:
    return {
        "summary": "Yellow painted housing with black hardware.",
        "reference_material_families": [],
        "current_render_findings": [],
        "matches": [],
        "mismatches": [],
        "consistency_issues": [],
        "assignment_guidance": [],
        "uncertainties": [],
    }


def test_observe_images_writes_structured_result_and_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "render.png"
    image.write_bytes(b"image")
    source_usd = tmp_path / "asset.usd"
    source_usd.write_bytes(b"usd")
    render_metadata_source = tmp_path / "render_metadata.json"
    render_metadata_source.write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    class FakeClient:
        last_token_usage = SimpleNamespace(
            to_dict=lambda: {
                "input_tokens": 120,
                "output_tokens": 30,
                "total_tokens": 150,
            }
        )

        def generate_with_image_caption_pairs(
            self,
            pairs: list[tuple[str, Path]],
            prompt: str,
            **kwargs: object,
        ) -> list[dict[str, str]]:
            observed["pairs"] = pairs
            observed["prompt"] = prompt
            observed["kwargs"] = kwargs
            return [
                {
                    "type": "text",
                    "text": json.dumps(_valid_observations()),
                }
            ]

    monkeypatch.setattr(vision_observer, "create_vlm", lambda **_kwargs: FakeClient())
    monkeypatch.setenv("TEST_VISION_KEY", "secret")
    output = tmp_path / "vision.json"

    result = vision_observer.observe_images(
        backend="test_vision_backend",
        model="openai/openai/gpt-5.6-sol",
        image_inputs=[{"label": "Current render", "path": str(image)}],
        phase="initial",
        output_path=output,
        api_key_env="TEST_VISION_KEY",
        source_usd=source_usd,
        render_metadata={
            str(image): {
                "direction": "+x-y+z",
                "ovrtx_render_mode": "rt2",
                "ovrtx_num_sensor_updates": 64,
            }
        },
        render_metadata_source=render_metadata_source,
    )

    assert result["parse_status"] == "parsed"
    assert result["observations"]["summary"].startswith("Yellow")
    assert isinstance(result["raw_response"], str)
    assert result["usage"]["total_tokens"] == 150
    assert result["source_usd"] == {
        "path": str(source_usd),
        "sha256": hashlib.sha256(b"usd").hexdigest(),
    }
    assert result["images"][0]["sha256"] == hashlib.sha256(b"image").hexdigest()
    assert result["images"][0]["render_metadata"]["ovrtx_render_mode"] == "rt2"
    assert result["render_metadata_source"] == {
        "path": str(render_metadata_source),
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }
    assert len(result["evidence_sha256"]) == 64
    assert observed["pairs"] == [("Current render", image)]
    assert "render failures" in str(observed["prompt"])
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_resolve_api_key_fails_when_named_environment_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_MISSING_VISION_KEY", raising=False)

    with pytest.raises(ValueError, match="environment variable is missing"):
        vision_observer._resolve_api_key(
            "test_vision_backend",
            "TEST_MISSING_VISION_KEY",
        )


def test_custom_base_url_requires_endpoint_scoped_api_key_environment() -> None:
    with pytest.raises(ValueError, match="custom vision base URL requires"):
        vision_observer._resolve_api_key(
            "test_vision_backend",
            None,
            base_url="https://custom.example.test/v1",
        )


def test_remote_vision_base_url_requires_https(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_VISION_KEY", "secret")

    with pytest.raises(ValueError, match="Invalid vision base URL"):
        vision_observer._resolve_api_key(
            "test_vision_backend",
            "TEST_VISION_KEY",
            base_url="https://:443",
        )

    with pytest.raises(ValueError, match="HTTPS is required"):
        vision_observer._resolve_api_key(
            "test_vision_backend",
            "TEST_VISION_KEY",
            base_url="http://vision.example.test/v1",
        )

    assert (
        vision_observer._resolve_api_key(
            "test_vision_backend",
            "TEST_VISION_KEY",
            base_url="http://127.0.0.1:9000/v1",
        )
        == "secret"
    )


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://user:secret@vision.example.test/v1",
        "https://vision.example.test/v1?api_key=secret",
        "https://vision.example.test/v1#secret",
    ],
)
def test_vision_base_url_rejects_embedded_credentials(
    endpoint_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_VISION_KEY", "secret")

    with pytest.raises(ValueError, match="Invalid vision base URL"):
        vision_observer._resolve_api_key(
            "test_vision_backend",
            "TEST_VISION_KEY",
            base_url=endpoint_url,
        )


def test_parser_rejects_invalid_max_tokens_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(vision_observer.VISION_MAX_TOKENS_ENV, "not-an-integer")

    with pytest.raises(
        SystemExit,
        match=f"{vision_observer.VISION_MAX_TOKENS_ENV} must be an integer",
    ):
        vision_observer.build_parser()


def test_parser_rejects_non_positive_max_tokens() -> None:
    with pytest.raises(SystemExit):
        vision_observer.build_parser().parse_args(
            [
                "--run-dir",
                ".",
                "--phase",
                "initial",
                "--images-json",
                "images.json",
                "--output",
                "observations.json",
                "--max-tokens",
                "-1",
            ]
        )


def test_cli_rejects_images_outside_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    manifest = run_dir / "images.json"
    manifest.write_text(
        json.dumps([{"label": "outside", "path": str(outside)}]),
        encoding="utf-8",
    )
    monkeypatch.setenv(vision_observer.VISION_BACKEND_ENV, "test_vision_backend")
    monkeypatch.setenv(vision_observer.VISION_MODEL_ENV, "openai/openai/gpt-5.6-sol")

    with pytest.raises(ValueError, match="must stay inside run directory"):
        vision_observer.main(
            [
                "--run-dir",
                str(run_dir),
                "--phase",
                "final-review",
                "--images-json",
                str(manifest),
                "--output",
                str(run_dir / "observations.json"),
            ]
        )


def test_parse_observations_accepts_fenced_json() -> None:
    payload = _valid_observations()
    parsed, status = vision_observer._parse_observations(
        f"```json\n{json.dumps(payload)}\n```"
    )

    assert status == "parsed"
    assert parsed == payload


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("", "response was empty"),
        ("not JSON", "not a valid JSON object"),
        ('{"summary": "incomplete"}', "missing required fields"),
        (
            json.dumps({**_valid_observations(), "uncertainties": "none"}),
            "fields must be arrays",
        ),
    ],
)
def test_parse_observations_fails_closed_on_invalid_evidence(
    response: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        vision_observer._parse_observations(response)


def test_next_observation_path_preserves_prior_calls(tmp_path: Path) -> None:
    first = tmp_path / "vision_observations_final.json"
    second = tmp_path / "vision_observations_final_02.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    assert vision_observer._next_observation_path(first) == (
        tmp_path / "vision_observations_final_03.json"
    )
