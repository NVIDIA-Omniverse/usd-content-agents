# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mock backends for simulate mode.

Provides mock VLM, Chat, and ImageEmbedding models that plug into the real
pipeline so that ``--simulate`` runs the exact same code paths as a real run,
just with deterministic, instant, no-network backends.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any

import numpy as np
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from world_understanding.functions.models.backends.registry import (
    register_chat_backend,
    register_vlm_backend,
)
from world_understanding.functions.models.image_embedding_models import (
    BaseImageEmbeddingModel,
)
from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json_material_names(text: str) -> list[str] | None:
    """Return a strict JSON ``material_names`` list embedded in *text*.

    Material prompts contain prose and output-schema examples around the material
    library payload, so the full prompt is not itself valid JSON. Scan each object
    boundary with :class:`json.JSONDecoder` and accept only a direct
    ``material_names: list[str]`` field. ``None`` means no current-contract payload
    was found; an empty list is a valid payload and must not fall through to legacy
    parsing.
    """
    decoder = json.JSONDecoder()
    search_from = 0
    while (object_start := text.find("{", search_from)) != -1:
        try:
            payload, object_end = decoder.raw_decode(text, object_start)
        except json.JSONDecodeError:
            search_from = object_start + 1
            continue

        if isinstance(payload, dict):
            material_names = payload.get("material_names")
            if isinstance(material_names, list) and all(
                isinstance(name, str) for name in material_names
            ):
                return material_names
        search_from = max(object_start + 1, object_end)
    return None


def _extract_material_names(text: str) -> list[str]:
    """Parse material names from current JSON or supported legacy prompts.

    Handles multiple formats:
    1. ``{"material_names": ["Aluminum", ...]}``
    2. ``- **Material name**: Aluminum\\n  **Description**: ...``
    3. Simple bulleted list: ``- Aluminum``
    4. Validate repair format: ``Valid material names:\\n"X", "Y", ...``
    """
    json_material_names = _extract_json_material_names(text)
    if json_material_names is not None:
        return json_material_names

    # First try structured format: "**Material name**: <name>"
    structured = re.findall(
        r"\*\*Material name\*\*:\s*(.+)",
        text,
        re.IGNORECASE,
    )
    if structured:
        return [n.strip() for n in structured if n.strip()]

    # Validate repair format: "Valid material names:\n<quoted, comma-separated>"
    val_match = re.search(
        r"Valid material names:\s*\n(.+?)(?:\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if val_match:
        quoted = re.findall(r'"([^"]+)"', val_match.group(1))
        if quoted:
            return [n.strip() for n in quoted if n.strip()]

    # Fallback: look for "Available materials:" block with simple bullets
    match = re.search(
        r"Available materials:\s*\n((?:\s*[-*]\s*.+\n?)+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        # Fallback: numbered list
        match = re.search(
            r"Available materials:\s*\n((?:\s*\d+[.)]\s*.+\n?)+)",
            text,
            re.IGNORECASE,
        )
    if not match:
        # Fallback: newline-separated after "Available materials:"
        match = re.search(
            r"Available materials:\s*\n((?:.*\S.*\n?)+)",
            text,
            re.IGNORECASE,
        )
    if not match:
        return []

    raw = match.group(1)
    names: list[str] = []
    for line in raw.strip().splitlines():
        # Strip bullet / numbering prefix
        cleaned = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        if cleaned:
            names.append(cleaned)
    return names


def _deterministic_pick(items: list[str], seed_text: str) -> str:
    """Pick an item deterministically using a hash of *seed_text*."""
    if not items:
        return "Steel Painted Gray"
    h = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    return items[h % len(items)]


def _looks_like_physics_prompt(prompt: str, system_prompt: str) -> bool:
    text = f"{system_prompt}\n{prompt}".lower()
    return (
        "physical_properties" in text
        or "component_type" in text
        or "physics propert" in text
    )


def _mock_physics_answer() -> dict[str, Any]:
    return {
        "classification": {
            "asset_type": "geometric shape",
            "component_type": "rigid_body",
            "component_name": "cube",
            "material": "plastic",
            "confidence": 0.99,
            "physical_properties": {
                "density": 1200.0,
                "estimated_mass_kg": 1.0,
                "static_friction": 0.4,
                "dynamic_friction": 0.32,
                "restitution": 0.4,
            },
        }
    }


# ---------------------------------------------------------------------------
# MockVLM
# ---------------------------------------------------------------------------


class MockVLM(BaseVisionLanguageModel):
    """Vision-Language Model that returns a deterministic material pick.

    Parses the ``Available materials:`` section from the system prompt,
    picks one deterministically (hash of prompt text), and returns a
    well-formed ``<reasoning>...<answer>`` response.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def model_name(self) -> str:
        return "mock"

    @property
    def backend_name(self) -> str:
        return "mock"

    def generate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        if _looks_like_physics_prompt(prompt, system_prompt):
            answer_json = json.dumps(_mock_physics_answer())
            return (
                f"<reasoning>Mock: selected deterministic physics properties</reasoning>"
                f"<answer>{answer_json}</answer>"
            )

        materials = _extract_material_names(system_prompt)
        if not materials:
            materials = _extract_material_names(prompt)
        picked = _deterministic_pick(materials, prompt)
        answer_json = json.dumps({"material": picked})
        return (
            f"<reasoning>Mock: selected based on context</reasoning>"
            f"<answer>{answer_json}</answer>"
        )

    async def agenerate(
        self,
        prompt: str,
        images: list[Any] | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        return await asyncio.to_thread(
            self.generate,
            prompt,
            images,
            system_prompt,
            temperature,
            max_tokens,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# MockChatModel
# ---------------------------------------------------------------------------


class MockChatModel(BaseChatModel):
    """LangChain chat model that inspects the prompt to produce context-aware
    mock responses for validate, harmonize, and reconcile steps.
    """

    def _generate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._messages_to_text(messages)
        response = self._route_response(text)
        message = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: Any,
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "mock"

    # -- internal helpers --

    @staticmethod
    def _messages_to_text(messages: Any) -> str:
        parts: list[str] = []
        for m in messages:
            if hasattr(m, "content"):
                parts.append(str(m.content))
            elif isinstance(m, dict) and "content" in m:
                parts.append(str(m["content"]))
        return "\n".join(parts)

    @staticmethod
    def _route_response(text: str) -> str:
        text_lower = text.lower()

        # --- Validate repair ---
        # Prompt typically lists invalid names and valid names, asks for a mapping.
        if "invalid" in text_lower and "valid" in text_lower:
            return MockChatModel._validate_response(text)

        # --- Harmonize ---
        if "conflict" in text_lower or "harmonize" in text_lower:
            return MockChatModel._harmonize_response(text)

        # --- Reconcile ---
        if "reconcile" in text_lower or "ambiguous" in text_lower:
            return MockChatModel._reconcile_response(text)

        # --- Default ---
        return '<answer>{"material": "Steel Painted Gray"}</answer>'

    @staticmethod
    def _validate_response(text: str) -> str:
        """Return a JSON mapping each invalid name to the first valid name found."""
        # Try to extract valid material names
        valid_names = _extract_material_names(text)
        first_valid = valid_names[0] if valid_names else "Steel Painted Gray"

        # Try to find invalid names (often quoted) without scanning the valid
        # material library or JSON keys as invalid-name candidates.
        invalid_section = re.search(
            r"Invalid names to fix:\s*\n(.+?)(?:\n\n|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        invalid_source = invalid_section.group(1) if invalid_section else text
        invalid_names = re.findall(r'"([^"]+)"', invalid_source)
        # Filter out names that are in the valid set
        valid_set = set(valid_names)
        invalids = [n for n in invalid_names if n not in valid_set and len(n) > 2]

        if invalids:
            mapping = dict.fromkeys(invalids[:10], first_valid)
            return f"<answer>{json.dumps(mapping)}</answer>"
        return f"<answer>{json.dumps({first_valid: first_valid})}</answer>"

    @staticmethod
    def _harmonize_response(text: str) -> str:
        """Return a unify action with the first material mentioned."""
        materials = _extract_material_names(text)
        mat = materials[0] if materials else "Steel Painted Gray"
        return f"<answer>{json.dumps({'action': 'unify', 'material': mat})}</answer>"

    @staticmethod
    def _reconcile_response(text: str) -> str:
        """Return a mapping from the first invalid to the first valid name."""
        valid_names = _extract_material_names(text)
        first_valid = valid_names[0] if valid_names else "Steel Painted Gray"
        return f"<answer>{json.dumps({first_valid: first_valid})}</answer>"


# ---------------------------------------------------------------------------
# MockImageEmbeddingModel
# ---------------------------------------------------------------------------


class MockImageEmbeddingModel(BaseImageEmbeddingModel):
    """Image embedding model returning deterministic 768-d vectors.

    Each image gets a slightly different vector (seeded from image content hash)
    so that downstream clustering produces real clusters.
    """

    AVAILABLE_MODELS = ["mock"]
    DEFAULT_MODEL = "mock"

    def __init__(self, **kwargs: Any) -> None:
        # Skip parent __init__ which creates an OpenAI client.
        self.api_key = "not-used"
        self.model = "mock"
        self.base_url = None
        self.timeout = 120.0
        self._embedding_dim = 768

    @classmethod
    def list_available_models(cls) -> list[str]:
        return cls.AVAILABLE_MODELS.copy()

    def embed_images(
        self,
        images: list[Any],
        **kwargs: Any,
    ) -> list[np.ndarray]:
        if not images:
            return []

        results: list[np.ndarray] = []
        for i, image in enumerate(images):
            seed = self._image_seed(image, i)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._embedding_dim).astype(np.float32)
            # Normalize to unit length
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            results.append(vec)
        return results

    @staticmethod
    def _image_seed(image: Any, index: int) -> int:
        """Derive a deterministic seed from an image."""
        if isinstance(image, str):
            data = image.encode()
        elif hasattr(image, "tobytes"):
            # PIL Image or numpy array — use first 1024 bytes for speed
            raw = image.tobytes()
            data = raw[:1024]
        else:
            data = str(index).encode()
        return int(hashlib.blake2s(data, digest_size=16).hexdigest(), 16) % (2**31)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _create_mock_chat(**kwargs: Any) -> BaseChatModel:
    return MockChatModel()


def _create_mock_vlm(**kwargs: Any) -> MockVLM:
    return MockVLM()


register_chat_backend("mock", _create_mock_chat, requires_api_key=False)
register_vlm_backend("mock", _create_mock_vlm, requires_api_key=False)
