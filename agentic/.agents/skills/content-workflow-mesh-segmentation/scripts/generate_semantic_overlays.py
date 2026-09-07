#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate recorded semantic overlays for arbitrary saved views."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mesh_geometry import sha256_file, write_json
from PIL import Image

_SUBPROCESS_STARTUP_GRACE_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-part", required=True)
    parser.add_argument("--exclude-part", action="append", default=[])
    parser.add_argument("--backend", default="openai_compatible")
    parser.add_argument("--model")
    parser.add_argument(
        "--base-url",
        help=(
            "Optional World Understanding image-backend base URL. The "
            "openai_compatible compatibility backend requires it."
        ),
    )
    parser.add_argument("--api-key-env")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--image-generation-tool",
        type=Path,
        help="Explicit path to the shared image-generation generate_image.py tool.",
    )
    return parser.parse_args()


def semantic_prompt(target_name: str, exclusions: list[str]) -> str:
    excluded = ", ".join(exclusions) if exclusions else "all neighboring parts"
    return f"""Edit the supplied neutral-gray 3D render as a semantic overlay.

Preserve the exact object, camera viewpoint, projection, framing, silhouette,
lighting, background, and every geometric edge. Do not redraw, move, scale,
crop, add, remove, beautify, or reinterpret geometry.

Tint every visible pixel belonging to the semantic part "{target_name}" with
one flat opaque chroma color: RGB (255, 0, 255), hex #FF00FF. Preserve original
pixels everywhere else. Include every visible target instance, including
partially occluded instances. Exclude {excluded}. At ambiguous boundaries,
prefer unchanged pixels over leaking magenta onto a neighbor. This is fallible
registration evidence, not a final segmentation."""


def _source_images(input_dir: Path) -> list[Path]:
    ignored_suffixes = (
        "_normal",
        "_linear_depth",
        "_face_ids",
        "_fragment_ids",
        "_label_ids",
        "_preview",
        "_blend",
    )
    paths = [
        path
        for path in sorted(input_dir.glob("*.png"))
        if not path.stem.endswith(ignored_suffixes)
    ]
    if not paths:
        raise FileNotFoundError(f"No neutral PNG renders found in {input_dir}")
    return paths


def _image_generation_tool(explicit: Path | None = None) -> Path:
    script_path = Path(__file__).resolve()
    configured = explicit or (
        Path(os.environ["WU_IMAGE_GENERATION_TOOL"])
        if os.getenv("WU_IMAGE_GENERATION_TOOL")
        else None
    )
    candidates = tuple(
        candidate
        for candidate in (
            configured,
            script_path.parents[2]
            / "image-generation"
            / "scripts"
            / "generate_image.py",
            script_path.parents[1]
            / ".agents"
            / "skills"
            / "image-generation"
            / "scripts"
            / "generate_image.py",
            Path("agentic/.agents/skills/image-generation/scripts/generate_image.py"),
            Path(".agents/skills/image-generation/scripts/generate_image.py"),
        )
        if candidate is not None
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "image-generation skill script is unavailable; pass "
        f"--image-generation-tool. Checked: {checked}"
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = semantic_prompt(args.target_part, list(args.exclude_part))
    prompt_path = output_dir / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    image_generation_tool = _image_generation_tool(args.image_generation_tool)
    records = []
    for source_path in _source_images(input_dir):
        output_path = output_dir / f"{source_path.stem}_raw_overlay.png"
        generation_manifest_path = (
            output_dir / f"{source_path.stem}_image_generation.json"
        )
        command = [
            sys.executable,
            str(image_generation_tool),
            "backend",
            "--prompt-file",
            str(prompt_path),
            "--conditioning-image",
            str(source_path),
            "--output",
            str(output_path),
            "--manifest",
            str(generation_manifest_path),
            "--backend",
            args.backend,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
        for option, value in (
            ("--model", args.model),
            ("--base-url", args.base_url),
            ("--api-key-env", args.api_key_env),
        ):
            if value:
                command.extend((option, value))
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds + _SUBPROCESS_STARTUP_GRACE_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Shared image generation exceeded its provider budget plus "
                f"startup grace for {source_path.stem}"
            ) from exc
        if completed.returncode != 0:
            detail = "shared image-generation attempt failed"
            if generation_manifest_path.is_file():
                failed_generation = json.loads(
                    generation_manifest_path.read_text(encoding="utf-8")
                )
                diagnostic = failed_generation.get("diagnostic")
                if isinstance(diagnostic, dict) and diagnostic.get("message"):
                    detail = str(diagnostic["message"])
            raise RuntimeError(
                f"Shared image generation failed for {source_path.stem}: {detail}"
            )
        generation_manifest = json.loads(
            generation_manifest_path.read_text(encoding="utf-8")
        )
        if generation_manifest.get("status") != "completed":
            raise RuntimeError(
                f"Shared image generation did not complete for {source_path.stem}"
            )
        with Image.open(source_path) as source_image:
            source_size = list(source_image.size)
        with Image.open(output_path) as generated:
            generated_size = list(generated.size)
        records.append(
            {
                "view_id": source_path.stem,
                "source_render": str(source_path),
                "source_render_sha256": sha256_file(source_path),
                "raw_overlay": str(output_path),
                "raw_overlay_sha256": sha256_file(output_path),
                "source_size": source_size,
                "generated_size": generated_size,
                "image_generation_manifest": str(generation_manifest_path),
                "image_generation_manifest_sha256": sha256_file(
                    generation_manifest_path
                ),
                "provider": generation_manifest["provider"],
            }
        )
        print(f"generated {source_path.stem}: {output_path}", flush=True)
    manifest = {
        "schema_version": "mesh-segmentation-semantic-overlays.v1",
        "target_semantic_part": args.target_part,
        "excluded_semantic_parts": list(args.exclude_part),
        "prompt": prompt,
        "backend": records[0]["provider"]["backend"] if records else args.backend,
        "model": records[0]["provider"]["model"] if records else args.model,
        "base_url": args.base_url,
        "transport": "agentic_image_generation_skill",
        "api_key_env": args.api_key_env,
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
