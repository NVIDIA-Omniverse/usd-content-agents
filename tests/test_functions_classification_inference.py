# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for classification inference hard timeouts."""

import asyncio
import json
import logging
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from PIL import Image as PILImage

import world_understanding.functions.classification.inference as inference_module
from world_understanding.functions.classification.inference import (
    _ainvoke_parser_with_chat_model,
    _call_async_with_timeout,
    _call_sync_with_timeout,
    _clean_unstructured_label,
    _explicit_unknown_sentinel_result,
    _explicit_unstructured_label_result,
    _extract_explicit_unstructured_label,
    _extract_image_metadata_from_entry,
    _extract_images_from_entry,
    _extract_text_from_entry,
    _get_vlm_generate_timeout_seconds,
    _invoke_parser_model_async,
    _invoke_parser_model_sync,
    _invoke_parser_with_chat_model,
    _is_exact_unknown_sentinel_text,
    _parse_multi_prim_response,
    _parse_single_result_from_response_text,
    _rename_legacy_material_key,
    _unwrap_transport_literal,
    async_classify_object,
    batch_classify_objects,
    classify_object,
    classify_objects_multi_prim,
    get_fibonacci_delay,
)
from world_understanding.utils.model_auth import (
    MODEL_AUTHENTICATION_FAILURE_MESSAGE,
    ModelAuthenticationFailure,
)
from world_understanding.utils.token_tracking import TokenTracker, TokenUsage


class _SlowVLM:
    """Minimal fake VLM that never responds before the deadline."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        time.sleep(0.05)
        return '{"class": "late"}'


class _AuthenticationError(RuntimeError):
    status_code = 401


def test_gpt5_parser_auth_failure_is_normalized() -> None:
    provider_body = "401 bearer-secret provider response and SDK internals"

    class _FailingGpt5Parser:
        model = "gpt-5-mini"

        def invoke(self, messages: list[Any], **kwargs: Any) -> NoReturn:
            raise _AuthenticationError(provider_body)

    with pytest.raises(ModelAuthenticationFailure) as exc_info:
        _invoke_parser_with_chat_model(
            _FailingGpt5Parser(),
            [],
            max_tokens=7,
        )

    assert str(exc_info.value) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert provider_body not in str(exc_info.value)


def test_parser_retry_auth_failure_is_normalized() -> None:
    provider_body = "401 bearer-secret retry response and SDK internals"

    class _RetryThenAuthParser:
        model = "parser"

        def invoke(self, messages: list[Any], **kwargs: Any) -> NoReturn:
            if "max_tokens" in kwargs:
                raise RuntimeError("max_tokens conflicts with max_completion_tokens")
            raise _AuthenticationError(provider_body)

    with pytest.raises(ModelAuthenticationFailure) as exc_info:
        _invoke_parser_with_chat_model(
            _RetryThenAuthParser(),
            [],
            max_tokens=7,
        )

    assert str(exc_info.value) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert provider_body not in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_parser_retry_auth_failure_is_normalized() -> None:
    provider_body = "401 bearer-secret async retry response and SDK internals"

    class _AsyncRetryThenAuthParser:
        async def ainvoke(self, messages: list[Any], **kwargs: Any) -> NoReturn:
            if "max_tokens" in kwargs:
                raise RuntimeError("max_tokens conflicts with max_completion_tokens")
            raise _AuthenticationError(provider_body)

    with pytest.raises(ModelAuthenticationFailure) as exc_info:
        await _ainvoke_parser_with_chat_model(
            _AsyncRetryThenAuthParser(),
            [],
            max_tokens=7,
        )

    assert str(exc_info.value) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert provider_body not in str(exc_info.value)


def test_classify_object_aborts_auth_failure_without_provider_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider_body = "401 bearer-secret provider response and SDK internals"

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(ModelAuthenticationFailure) as exc_info,
    ):
        classify_object(
            vlm=_SequenceVLM([_AuthenticationError(provider_body)]),
            text="classify it",
            images=["front.png"],
            llm=object(),
            max_retries=1,
        )

    assert str(exc_info.value) == MODEL_AUTHENTICATION_FAILURE_MESSAGE
    assert provider_body not in caplog.text


def test_parallel_batch_cancels_queued_work_after_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = Event()
    sibling_release = Event()
    calls: list[str] = []

    def fake_classify_object(**kwargs: Any) -> dict[str, str]:
        text = kwargs["text"]
        calls.append(text)
        if text == "auth":
            assert sibling_started.wait(timeout=1)
            raise _AuthenticationError("401 bearer-secret provider response")
        if text == "sibling":
            sibling_started.set()
            sibling_release.wait(timeout=0.2)
            return {"class": "finished"}
        raise AssertionError(f"queued work executed after auth failure: {text}")

    monkeypatch.setattr(inference_module, "classify_object", fake_classify_object)

    with pytest.raises(ModelAuthenticationFailure):
        batch_classify_objects(
            vlm=object(),
            entries=[
                {"id": "auth", "text": "auth"},
                {"id": "sibling", "text": "sibling"},
                *(
                    {"id": f"queued-{index}", "text": f"queued-{index}"}
                    for index in range(5)
                ),
            ],
            llm=object(),
            max_workers=2,
            max_retries=1,
        )

    assert set(calls) == {"auth", "sibling"}


@pytest.mark.asyncio
async def test_async_batch_cancels_and_drains_siblings_after_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = asyncio.Event()
    never_finish = asyncio.Event()
    calls: list[str] = []
    cancelled: list[str] = []

    async def fake_async_classify_object(**kwargs: Any) -> NoReturn:
        text = kwargs["text"]
        calls.append(text)
        if text == "auth":
            await sibling_started.wait()
            raise _AuthenticationError("401 bearer-secret provider response")
        if text == "sibling":
            sibling_started.set()
            try:
                await never_finish.wait()
            except asyncio.CancelledError:
                cancelled.append(text)
                raise
        raise AssertionError(f"queued work executed after auth failure: {text}")

    monkeypatch.setattr(
        inference_module, "async_classify_object", fake_async_classify_object
    )

    with pytest.raises(ModelAuthenticationFailure):
        await inference_module.async_batch_classify_objects(
            vlm=object(),
            entries=[
                {"id": "auth", "text": "auth"},
                {"id": "sibling", "text": "sibling"},
                {"id": "queued", "text": "queued"},
            ],
            llm=object(),
            max_workers=2,
            max_retries=1,
        )

    assert set(calls) == {"auth", "sibling"}
    assert cancelled == ["sibling"]


class _ParserFallbackVLM:
    """Fake VLM that needs a second text-only parser pass to return JSON."""

    last_token_usage = None

    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("images"):
            return "The best match is metal."
        return '{"class": "metal"}'

    async def agenerate(self, *args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("images"):
            return "The best match is metal."
        return '{"class": "metal"}'


class _JsonThenPlaceholderAnswerVLM:
    """Fake VLM that emits valid JSON plus a stale answer placeholder."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return """<reasoning>
The part is a black structural shoulder component.
</reasoning>

```json
{
  "material": "Steel Painted Black"
}
```
<answer>your answer</answer>"""


class _PhysicsStructuredAnswerVLM:
    """Fake VLM that emits the physics-agent structured answer schema."""

    last_token_usage = None

    @staticmethod
    def _response():
        payload = {
            "asset_type": "light fixture",
            "component_type": "housing",
            "component_name": "lamp housing",
            "material": "metal",
            "physical_properties": {
                "density": 7850,
                "estimated_mass_kg": 0.4,
                "static_friction": 0.6,
                "dynamic_friction": 0.45,
                "restitution": 0.1,
            },
            "confidence": "high",
            "reasoning": "Visible metallic shell around the bulb.",
        }
        return f"<answer>\n{json.dumps(payload)}\n</answer>"

    def generate(self, *args, **kwargs):
        return self._response()

    async def agenerate(self, *args, **kwargs):
        return self._response()


class _FencedPhysicsStructuredAnswerVLM:
    """Fake VLM that emits fenced physics-agent JSON without answer tags."""

    last_token_usage = None

    @staticmethod
    def _response():
        payload = {
            "asset_type": "office chair",
            "component_type": "mechanical",
            "component_name": "wheel",
            "material": "plastic",
            "physical_properties": {
                "density": 1200,
                "estimated_mass_kg": 0.156,
                "static_friction": 0.4,
                "dynamic_friction": 0.3,
                "restitution": 0.4,
            },
            "confidence": "high",
            "reasoning": "Office chair casters are typically durable plastic.",
        }
        return (
            "<reasoning>\n"
            "The highlighted component is an office chair caster wheel.\n"
            "</reasoning>\n"
            "```json\n"
            f"{json.dumps(payload, indent=2)}\n"
            "```"
        )

    def generate(self, *args, **kwargs):
        return self._response()

    async def agenerate(self, *args, **kwargs):
        return self._response()


class _MultipleFencedLegacyMaterialVLM:
    """Fake VLM that emits misleading JSON before the final legacy answer."""

    last_token_usage = None

    @staticmethod
    def _response():
        example_payload = {
            "asset_type": "example",
            "component_type": "sample",
            "material": "metal",
        }
        final_payload = {
            "asset_type": "office chair",
            "component_type": "mechanical",
            "component_name": "wheel",
            "material": "plastic",
            "physical_properties": {
                "density": 1200,
                "estimated_mass_kg": 0.156,
            },
            "confidence": "high",
        }
        return (
            "The prompt included this example:\n"
            "```json\n"
            f"{json.dumps(example_payload, indent=2)}\n"
            "```\n"
            "Intermediate reasoning:\n"
            "```json\n"
            '{"thinking": "caster wheels are often plastic"}\n'
            "```\n"
            "Final answer:\n"
            "```json\n"
            f"{json.dumps(final_payload, indent=2)}\n"
            "```"
        )

    def generate(self, *args, **kwargs):
        return self._response()

    async def agenerate(self, *args, **kwargs):
        return self._response()


class _StructuredAnswerWithRequestedOutputKeyVLM:
    """Fake VLM that emits both the requested output key and material metadata."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        payload = {
            "classification": "fixture housing",
            "material": "metal",
            "physical_properties": {"density": 7850},
        }
        return f"<answer>\n{json.dumps(payload)}\n</answer>"


class _RequestedKeyBeforeLegacyMaterialVLM:
    """Fake VLM that emits an explicit requested key before legacy material JSON."""

    last_token_usage = None

    @staticmethod
    def _response():
        requested_payload = {
            "classification": "caster wheel",
            "confidence": "medium",
        }
        legacy_payload = {
            "material": "plastic",
            "confidence": "high",
        }
        return (
            "```json\n"
            f"{json.dumps(requested_payload, indent=2)}\n"
            "```\n"
            "Additional legacy material estimate:\n"
            "```json\n"
            f"{json.dumps(legacy_payload, indent=2)}\n"
            "```"
        )

    def generate(self, *args, **kwargs):
        return self._response()

    async def agenerate(self, *args, **kwargs):
        return self._response()


class _ReasoningOnlyClassificationVLM:
    """Fake VLM that states a class plainly but never emits JSON."""

    last_token_usage = None

    @staticmethod
    def _response():
        return (
            "<reasoning>\n"
            "The overall object is a utility cart with drawers and wheels.\n"
            "The isolated component consists of two long rails attached to the "
            "cart frame.\n"
            "Given the strict classification rules for articulated props, "
            'these rails must be classified as "body". Since the rails are '
            'rigidly attached, they fall under the "body" category.\n'
        )

    def generate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()

    async def agenerate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()


class _NegatedReasoningOnlyClassificationVLM:
    """Fake VLM with a negated label mention before the final choice."""

    last_token_usage = None

    @staticmethod
    def _response():
        return (
            "<reasoning>\n"
            'The rails should not be classified as "body" if they slide. '
            'In this case they move independently, so they must be classified as "joint".'
        )

    def generate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()

    async def agenerate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()


class _MultiwordReasoningOnlyClassificationVLM:
    """Fake VLM with conjunctions and prepositions inside the class label."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return '<reasoning>\nThe final answer is "black and white adapter for rail".'


class _TrailingWordReasoningOnlyClassificationVLM:
    """Fake VLM whose valid label ends with a category-like word."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return '<reasoning>\nThe final classification is "safety category".'


class _ReasoningOnlyUnknownClassificationVLM:
    """Fake VLM that states an unknown sentinel in unstructured text."""

    last_token_usage = None

    @staticmethod
    def _response():
        return "<reasoning>\nThe final classification is Unknown."

    def generate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()

    async def agenerate(self, *args, **kwargs):
        if kwargs.get("images") is None:
            return "I could not format this response."
        return self._response()


class _UnknownSentinelVLM:
    """Fake VLM that returns an unstructured unknown sentinel."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return "__UNKNOWN__"

    async def agenerate(self, *args, **kwargs):
        return "__UNKNOWN__"


class _QuotedUnknownSentinelVLM:
    """Fake VLM that returns a quoted sentinel literal."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return '"__UNKNOWN__"'


class _AnswerUnknownSentinelVLM:
    """Fake VLM that returns the sentinel in a non-JSON answer block."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return "<answer>__UNKNOWN__</answer>"

    async def agenerate(self, *args, **kwargs):
        return "<answer>__UNKNOWN__</answer>"


class _NegatedUnknownJsonVLM:
    """Fake VLM that mentions the sentinel while choosing a concrete material."""

    last_token_usage = None

    def generate(self, *args, **kwargs):
        return '{"material": "Not __UNKNOWN__, it is Steel"}'

    async def agenerate(self, *args, **kwargs):
        return '{"material": "Not __UNKNOWN__, it is Steel"}'


class _MaterialParserLLM:
    """Fake parser LLM that returns a concrete material."""

    def invoke(self, *args, **kwargs):
        class Response:
            content = '{"material": "Steel"}'

        return Response()


def test_classify_object_enforces_hard_timeout(monkeypatch):
    """Slow VLM calls should fail fast instead of hanging indefinitely."""
    monkeypatch.setenv("WU_VLM_GENERATE_TIMEOUT_SECONDS", "0.01")

    with pytest.raises(TimeoutError, match="VLM generate did not respond"):
        classify_object(
            vlm=_SlowVLM(),
            text="classify this",
            images=["unused.png"],
            llm=object(),
            max_retries=1,
        )


def test_classify_object_supports_vlm_parser_fallback():
    """The parser fallback should support llm=vlm without requiring .invoke()."""
    parser_vlm = _ParserFallbackVLM()

    result = classify_object(
        vlm=parser_vlm,
        text="classify this object",
        images=["unused.png"],
        llm=parser_vlm,
        max_retries=1,
    )

    assert result["class"] == "metal"
    assert len(parser_vlm.calls) == 2
    assert parser_vlm.calls[0]["images"] == ["unused.png"]
    assert parser_vlm.calls[1]["images"] is None


def test_classify_object_prefers_full_response_json_over_placeholder_answer():
    """A stale answer placeholder should not override valid JSON in the response."""
    result = classify_object(
        vlm=_JsonThenPlaceholderAnswerVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
    )

    assert result["material"] == "Steel Painted Black"
    assert result["original_response"].endswith("<answer>your answer</answer>")


def test_classify_object_preserves_structured_answer_json_for_custom_output_key():
    """Structured VLM answer JSON should not collapse to a single string."""
    result = classify_object(
        vlm=_PhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "metal"
    assert result["asset_type"] == "light fixture"
    assert result["component_type"] == "housing"
    assert result["component_name"] == "lamp housing"
    assert result["physical_properties"]["density"] == 7850
    assert result["confidence"] == "high"
    assert result["reasoning"] == "Visible metallic shell around the bulb."
    assert result["original_response"].startswith("<answer>")
    assert "material" not in result


def test_classify_object_parses_fenced_structured_json_for_custom_output_key():
    """Fenced legacy material JSON should normalize to the requested key."""
    result = classify_object(
        vlm=_FencedPhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "plastic"
    assert result["asset_type"] == "office chair"
    assert result["component_type"] == "mechanical"
    assert result["component_name"] == "wheel"
    assert result["physical_properties"]["density"] == 1200
    assert result["confidence"] == "high"
    assert result["original_response"].startswith("<reasoning>")
    assert "material" not in result


def test_classify_object_prefers_later_legacy_material_json_candidate():
    """Earlier example JSON should not mask a later legacy material answer."""
    result = classify_object(
        vlm=_MultipleFencedLegacyMaterialVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "plastic"
    assert result["asset_type"] == "office chair"
    assert result["component_type"] == "mechanical"
    assert result["component_name"] == "wheel"
    assert result["physical_properties"]["density"] == 1200
    assert "material" not in result


def test_classify_object_preserves_material_when_output_key_already_exists():
    """A sibling material field should not overwrite the requested output key."""
    result = classify_object(
        vlm=_StructuredAnswerWithRequestedOutputKeyVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "fixture housing"
    assert result["material"] == "metal"
    assert result["physical_properties"]["density"] == 7850


def test_classify_object_prefers_requested_key_over_later_legacy_material():
    """Explicit output-key JSON should win over a later legacy material block."""
    result = classify_object(
        vlm=_RequestedKeyBeforeLegacyMaterialVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "caster wheel"
    assert result["confidence"] == "medium"
    assert "material" not in result


def test_classify_object_extracts_explicit_reasoning_only_classification():
    """Reasoning-only VLM text should not collapse to Unable to parse."""
    result = classify_object(
        vlm=_ReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_ReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "body"
    assert "must be classified" in result["original_response"]


def test_classify_object_ignores_negated_unstructured_classification():
    """Negated label mentions should not win over later affirmative labels."""
    result = classify_object(
        vlm=_NegatedReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_NegatedReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "joint"


def test_classify_object_preserves_conjunctions_inside_unstructured_label():
    """Words like 'and' and 'for' can be part of a valid label."""
    result = classify_object(
        vlm=_MultiwordReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_MultiwordReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "black and white adapter for rail"


def test_classify_object_preserves_category_like_unstructured_label_suffix():
    """Words like 'category' can be part of a valid label."""
    result = classify_object(
        vlm=_TrailingWordReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_TrailingWordReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "safety category"


def test_classify_object_preserves_unknown_sentinel_without_parser_fallback():
    """An explicit configured sentinel should not be replaced by LLM guessing."""
    result = classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"
    assert "__UNKNOWN__" in result["original_response"]


def test_classify_object_canonicalizes_unstructured_unknown_sentinel():
    """Sync unstructured label extraction should preserve sentinel casing."""
    result = classify_object(
        vlm=_ReasoningOnlyUnknownClassificationVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="unknown",
    )

    assert result["material"] == "unknown"
    assert "classification is Unknown" in result["original_response"]


def test_classify_object_canonicalizes_sentinel_answer_block():
    """Answer blocks with the exact configured sentinel should become sentinel."""
    result = classify_object(
        vlm=_AnswerUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"


def test_classify_object_does_not_canonicalize_negated_sentinel_value():
    """JSON values that merely reference the sentinel should stay intact."""
    result = classify_object(
        vlm=_NegatedUnknownJsonVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "Not __UNKNOWN__, it is Steel"


def test_classify_object_preserves_configured_quoted_sentinel_literal():
    """A quoted sentinel config should not be stripped during comparison."""
    result = classify_object(
        vlm=_QuotedUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel='"__UNKNOWN__"',
    )

    assert result["material"] == '"__UNKNOWN__"'


def test_classify_object_does_not_strip_quotes_from_configured_sentinel():
    """An unquoted VLM value should not match a quoted configured sentinel."""
    result = classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=_MaterialParserLLM(),
        output_key="material",
        max_retries=1,
        unknown_sentinel='"__UNKNOWN__"',
    )

    assert result["material"] == "Steel"


@pytest.mark.asyncio
async def test_async_classify_object_supports_vlm_parser_fallback():
    """Async classification should support llm=vlm without requiring .ainvoke()."""
    parser_vlm = _ParserFallbackVLM()

    result = await async_classify_object(
        vlm=parser_vlm,
        text="classify this object",
        images=["unused.png"],
        llm=parser_vlm,
        max_retries=1,
    )

    assert result["class"] == "metal"
    assert len(parser_vlm.calls) == 2
    assert parser_vlm.calls[0]["images"] == ["unused.png"]
    assert parser_vlm.calls[1]["images"] is None


@pytest.mark.asyncio
async def test_async_classify_object_preserves_structured_answer_json_for_custom_output_key():
    """Async classification should preserve structured VLM answer JSON too."""
    result = await async_classify_object(
        vlm=_PhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "metal"
    assert result["asset_type"] == "light fixture"
    assert result["component_type"] == "housing"
    assert result["component_name"] == "lamp housing"
    assert result["physical_properties"]["density"] == 7850
    assert result["confidence"] == "high"
    assert result["reasoning"] == "Visible metallic shell around the bulb."
    assert result["original_response"].startswith("<answer>")
    assert "material" not in result


@pytest.mark.asyncio
async def test_async_classify_object_parses_fenced_structured_json_for_custom_output_key():
    """Async classification should normalize fenced legacy material JSON too."""
    result = await async_classify_object(
        vlm=_FencedPhysicsStructuredAnswerVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "plastic"
    assert result["asset_type"] == "office chair"
    assert result["component_type"] == "mechanical"
    assert result["component_name"] == "wheel"
    assert result["physical_properties"]["density"] == 1200
    assert result["confidence"] == "high"
    assert result["original_response"].startswith("<reasoning>")
    assert "material" not in result


@pytest.mark.asyncio
async def test_async_classify_object_prefers_later_legacy_material_json_candidate():
    """Async fallback should keep scanning to the final legacy material JSON."""
    result = await async_classify_object(
        vlm=_MultipleFencedLegacyMaterialVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "plastic"
    assert result["asset_type"] == "office chair"
    assert result["component_type"] == "mechanical"
    assert result["component_name"] == "wheel"
    assert result["physical_properties"]["density"] == 1200
    assert "material" not in result


@pytest.mark.asyncio
async def test_async_classify_object_prefers_requested_key_over_later_legacy_material():
    """Async explicit output-key JSON should win over later legacy material."""
    result = await async_classify_object(
        vlm=_RequestedKeyBeforeLegacyMaterialVLM(),
        text="classify this mechanical part",
        images=["unused.png"],
        llm=object(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "caster wheel"
    assert result["confidence"] == "medium"
    assert "material" not in result


@pytest.mark.asyncio
async def test_async_classify_object_extracts_explicit_reasoning_only_classification():
    """Async reasoning-only VLM text should not collapse to Unable to parse."""
    result = await async_classify_object(
        vlm=_ReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_ReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "body"
    assert "must be classified" in result["original_response"]


@pytest.mark.asyncio
async def test_async_classify_object_ignores_negated_unstructured_classification():
    """Async negated label mentions should not win over affirmative labels."""
    result = await async_classify_object(
        vlm=_NegatedReasoningOnlyClassificationVLM(),
        text="classify this articulated prop component",
        images=["unused.png"],
        llm=_NegatedReasoningOnlyClassificationVLM(),
        output_key="classification",
        max_retries=1,
    )

    assert result["classification"] == "joint"


@pytest.mark.asyncio
async def test_async_classify_object_preserves_unknown_sentinel_without_fallback():
    """Async single-object classification should preserve explicit sentinels too."""
    result = await async_classify_object(
        vlm=_UnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"
    assert "__UNKNOWN__" in result["original_response"]


@pytest.mark.asyncio
async def test_async_classify_object_canonicalizes_unstructured_unknown_sentinel():
    """Async unstructured label extraction should preserve sentinel casing."""
    result = await async_classify_object(
        vlm=_ReasoningOnlyUnknownClassificationVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="unknown",
    )

    assert result["material"] == "unknown"
    assert "classification is Unknown" in result["original_response"]


@pytest.mark.asyncio
async def test_async_classify_object_canonicalizes_sentinel_answer_block():
    """Async parser should canonicalize exact sentinel answer text."""
    result = await async_classify_object(
        vlm=_AnswerUnknownSentinelVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "__UNKNOWN__"


@pytest.mark.asyncio
async def test_async_classify_object_does_not_canonicalize_negated_sentinel_value():
    """Async JSON values that merely reference the sentinel should stay intact."""
    result = await async_classify_object(
        vlm=_NegatedUnknownJsonVLM(),
        text="classify this material",
        images=["unused.png"],
        llm=object(),
        output_key="material",
        max_retries=1,
        unknown_sentinel="__UNKNOWN__",
    )

    assert result["material"] == "Not __UNKNOWN__, it is Steel"


class _Response:
    def __init__(self, content):
        self.content = content


class _RetryingChatParser:
    def __init__(
        self,
        content='{"class": "parsed"}',
        *,
        model="parser",
        fail_max_tokens_once=False,
        async_fail_max_tokens_once=False,
    ):
        self.content = content
        self.model = model
        self.fail_max_tokens_once = fail_max_tokens_once
        self.async_fail_max_tokens_once = async_fail_max_tokens_once
        self.calls = []
        self.async_calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(kwargs)
        if self.fail_max_tokens_once and "max_tokens" in kwargs:
            self.fail_max_tokens_once = False
            raise RuntimeError("max_tokens conflicts with max_completion_tokens")
        return _Response(self.content)

    async def ainvoke(self, messages, **kwargs):
        self.async_calls.append(kwargs)
        if self.async_fail_max_tokens_once and "max_tokens" in kwargs:
            self.async_fail_max_tokens_once = False
            raise RuntimeError("max_tokens conflicts with max_completion_tokens")
        return _Response(self.content)


class _SyncOnlyChatParser:
    def __init__(self, content='{"class": "sync"}'):
        self.content = content
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.content)


class _SequenceVLM:
    def __init__(self, responses, *, token_usage=None):
        self.responses = list(responses)
        self.calls = []
        self.pair_calls = []
        self.async_calls = []
        self.async_pair_calls = []
        self.last_token_usage = token_usage

    def _next(self):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self._next()

    def generate_with_image_caption_pairs(self, *args, **kwargs):
        self.pair_calls.append(kwargs)
        return self._next()

    async def agenerate(self, *args, **kwargs):
        self.async_calls.append(kwargs)
        return self._next()

    async def agenerate_with_image_caption_pairs(self, *args, **kwargs):
        self.async_pair_calls.append(kwargs)
        return self._next()


def test_parser_and_timeout_helpers_cover_edge_branches(monkeypatch):
    assert get_fibonacci_delay(-1, base_delay=2.0) == 2.0
    assert get_fibonacci_delay(1, base_delay=3.0) == 3.0
    assert get_fibonacci_delay(5, base_delay=0.5) == 4.0

    monkeypatch.setenv("WU_VLM_GENERATE_TIMEOUT_SECONDS", "not-a-number")
    assert _get_vlm_generate_timeout_seconds() == 180.0
    monkeypatch.setenv("WU_VLM_GENERATE_TIMEOUT_SECONDS", "-3")
    assert _get_vlm_generate_timeout_seconds() == 180.0

    assert (
        _call_sync_with_timeout(
            lambda: "ok", timeout_seconds=1.0, operation_name="sync op"
        )
        == "ok"
    )

    async def _async_ok():
        return "async-ok"

    assert (
        asyncio.run(
            _call_async_with_timeout(
                _async_ok(), timeout_seconds=1.0, operation_name="async op"
            )
        )
        == "async-ok"
    )

    async def _async_slow():
        await asyncio.sleep(0.01)
        return "late"

    with pytest.raises(TimeoutError, match="async slow did not respond"):
        asyncio.run(
            _call_async_with_timeout(
                _async_slow(), timeout_seconds=0.001, operation_name="async slow"
            )
        )

    gpt5_parser = _RetryingChatParser(model="gpt-5-mini")
    assert _invoke_parser_with_chat_model(gpt5_parser, [], max_tokens=7) == (
        '{"class": "parsed"}'
    )
    assert gpt5_parser.calls == [{"max_completion_tokens": 7}]

    retry_parser = _RetryingChatParser(
        content=123,
        fail_max_tokens_once=True,
    )
    assert _invoke_parser_with_chat_model(retry_parser, [], max_tokens=9) == "123"
    assert retry_parser.calls == [
        {"temperature": 0.1, "max_tokens": 9},
        {"max_completion_tokens": 9},
    ]

    async_retry_parser = _RetryingChatParser(async_fail_max_tokens_once=True)
    assert (
        asyncio.run(
            _ainvoke_parser_with_chat_model(async_retry_parser, [], max_tokens=5)
        )
        == '{"class": "parsed"}'
    )
    assert async_retry_parser.async_calls == [
        {"temperature": 0.1, "max_tokens": 5},
        {"max_completion_tokens": 5},
    ]

    sync_only = _SyncOnlyChatParser()
    assert (
        asyncio.run(_ainvoke_parser_with_chat_model(sync_only, [], max_tokens=3))
        == '{"class": "sync"}'
    )
    assert sync_only.calls == [{"temperature": 0.1, "max_tokens": 3}]

    with pytest.raises(TypeError, match="Parser model must support"):
        _invoke_parser_model_sync(
            object(),
            messages=[],
            parsing_prompt="prompt",
            parser_system_prompt="system",
            max_tokens=1,
        )
    with pytest.raises(TypeError, match="Parser model must support"):
        asyncio.run(
            _invoke_parser_model_async(
                object(),
                messages=[],
                parsing_prompt="prompt",
                parser_system_prompt="system",
                max_tokens=1,
            )
        )


def test_single_result_parsing_helpers_cover_fallbacks():
    assert _parse_single_result_from_response_text("", output_key="class") is None
    assert _parse_single_result_from_response_text("   ", output_key="class") is None

    multi_answer = "<answer>example</answer><answer>plain text</answer>"
    assert _parse_single_result_from_response_text(
        multi_answer, output_key="class"
    ) == {"class": "plain text"}

    list_answer = "<answer>[1, 2]</answer>"
    assert _parse_single_result_from_response_text(list_answer, output_key="class") == {
        "class": "[1, 2]"
    }

    dict_without_value = '<answer>{"note": "not a class"}</answer>'
    assert _parse_single_result_from_response_text(
        dict_without_value, output_key="class"
    ) == {"class": '{"note": "not a class"}'}

    legacy = {"material": "steel"}
    _rename_legacy_material_key(legacy, output_key="classification", value="steel")
    assert legacy == {"classification": "steel"}

    already_has_key = {"classification": "wheel", "material": "rubber"}
    _rename_legacy_material_key(
        already_has_key, output_key="classification", value="rubber"
    )
    assert already_has_key == {"classification": "wheel", "material": "rubber"}

    material_result = {}
    _rename_legacy_material_key(material_result, output_key="material", value="wood")
    assert material_result == {"material": "wood"}

    assert _extract_explicit_unstructured_label("") is None
    assert (
        _extract_explicit_unstructured_label("This is not classified as body.") is None
    )
    assert (
        _extract_explicit_unstructured_label(
            "First classified as body. Later the final label is wheel."
        )
        == "wheel"
    )
    assert _clean_unstructured_label("   !!!   ") is None
    assert _clean_unstructured_label("x" * 81) is None
    assert _clean_unstructured_label('"steel because it shines"') == "steel"

    assert (
        _explicit_unknown_sentinel_result(
            "", output_key="class", unknown_sentinel="unknown"
        )
        is None
    )
    assert not _is_exact_unknown_sentinel_text("unknown", None)
    assert not _is_exact_unknown_sentinel_text("unknown", "   ")
    assert _is_exact_unknown_sentinel_text("'unknown'", "unknown")
    assert not _is_exact_unknown_sentinel_text("unknown", '"unknown"')
    assert _unwrap_transport_literal("plain") == "plain"

    assert _explicit_unstructured_label_result(
        "final classification is Unknown",
        output_key="class",
        unknown_sentinel="unknown",
    ) == {"class": "unknown"}


def test_classify_object_retries_pairs_tracks_tokens_and_fallbacks(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    tracker = TokenTracker()
    usage = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    vlm = _SequenceVLM(
        ["", '{"class": "chair"}'],
        token_usage=usage,
    )

    result = classify_object(
        vlm=vlm,
        text="classify it",
        images=["front.png"],
        llm=object(),
        invoke_kwargs={"temperature": 0.2, "max_completion_tokens": 4},
        image_prompts=["front view"],
        max_retries=2,
    )

    assert result["class"] == "chair"
    assert [call["max_tokens"] for call in vlm.pair_calls] == [4, 8]
    assert tracker.get_stats()["invocation_count"] == 0

    vlm_with_tracker = _SequenceVLM(
        ['{"class": "table"}'],
        token_usage=usage,
    )
    tracked = TokenTracker()
    assert (
        classify_object(
            vlm=vlm_with_tracker,
            text="classify it",
            images=["front.png"],
            llm=object(),
            token_tracker=tracked,
            max_retries=1,
        )["class"]
        == "table"
    )
    assert tracked.get_stats()["invocation_count"] == 1

    fallback = classify_object(
        vlm=_SequenceVLM(["<answer>loose answer</answer>"]),
        text="classify it",
        images=["front.png"],
        llm=object(),
        image_prompts=["front", "extra"],
        max_retries=1,
    )
    assert fallback["class"] == "loose answer"

    retry_after_error = classify_object(
        vlm=_SequenceVLM([RuntimeError("temporary"), '{"class": "after-error"}']),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=2,
    )
    assert retry_after_error["class"] == "after-error"


def test_classify_object_parser_fallback_failure_modes(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    unable = classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_RetryingChatParser(content="not json"),
        max_retries=1,
    )
    assert unable == {
        "class": "Unable to parse",
        "original_response": "unstructured and undecidable",
    }

    class _FailingParser:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("parser down")

    failed = classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_FailingParser(),
        max_retries=1,
    )
    assert failed == {
        "class": "Error during parsing",
        "original_response": "unstructured and undecidable",
    }

    empty_then_parsed = classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_SequenceVLM(["", '{"class": "fallback"}']),
        max_retries=2,
    )
    assert empty_then_parsed["class"] == "fallback"


def test_batch_classify_objects_sequential_branches(tmp_path):
    image_path = tmp_path / "front.png"
    image_path.write_bytes(b"not really an image")

    events = SimpleNamespace(progress=[], errors=[], results=[], predictions=[])

    def on_result(result, entry):
        events.results.append((result["id"], result["status"]))
        if entry.get("raise_result"):
            raise RuntimeError("result callback failed")

    def on_prediction(entry_id, response):
        events.predictions.append((entry_id, response["class"]))
        raise RuntimeError("prediction callback failed")

    vlm = _SequenceVLM(['{"class": "good"}', '{"class": "pil"}'])
    pil_image = PILImage.new("RGB", (1, 1))

    results = batch_classify_objects(
        vlm=vlm,
        entries=[
            {"id": "skip", "images": ["front.png"], "text": "skip"},
            {"id": "bad", "images": [123, "missing.png"], "raise_result": True},
            {
                "id": "good",
                "images": ["front.png"],
                "text": "old prompt",
                "image_metadata": [{"vlm_prompt": "front caption"}],
            },
            {
                "id": "pil",
                "media": {"images": [pil_image]},
                "user_prompt": "new prompt",
                "system_prompt": "entry system",
            },
        ],
        llm=object(),
        image_base_dir=tmp_path,
        processed_ids={"skip"},
        on_progress=lambda entry_id, response: events.progress.append(entry_id),
        on_error=lambda entry_id, error: events.errors.append((entry_id, error)),
        on_result=on_result,
        on_prediction=on_prediction,
        max_retries=1,
    )

    assert [result["id"] for result in results] == ["bad", "good", "pil"]
    assert results[0]["status"] == "error"
    assert "Unsupported image types" in results[0]["error"]
    assert "Missing images" in results[0]["error"]
    assert results[1]["vlm_response"]["class"] == "good"
    assert results[2]["vlm_response"]["class"] == "pil"
    assert events.progress == ["good", "pil"]
    assert [item[0] for item in events.errors] == ["bad"]
    assert events.predictions == [("good", "good"), ("pil", "pil")]
    assert len(vlm.pair_calls) == 1
    assert len(vlm.calls) == 1

    assert _extract_images_from_entry({"image_path": "one.png"}) == ["one.png"]
    assert _extract_images_from_entry({"media": {"images": [{"path": "two.png"}]}}) == [
        "two.png"
    ]
    assert _extract_images_from_entry({}) == []
    assert _extract_image_metadata_from_entry({"image_metadata": [{"a": 1}]}) == [
        {"a": 1}
    ]
    assert _extract_image_metadata_from_entry({}) == []
    assert _extract_text_from_entry({"user_prompt": "hello"}) == "hello"
    assert _extract_text_from_entry({}) == ""


def test_batch_classify_objects_parallel_branches(monkeypatch, tmp_path):
    image_path = tmp_path / "front.png"
    image_path.write_bytes(b"not really an image")
    events = SimpleNamespace(progress=[], errors=[], results=[], predictions=[])

    def fake_classify_object(**kwargs):
        if "explode" in kwargs["text"]:
            raise RuntimeError("classification exploded")
        return {"class": f"ok-{kwargs['text']}"}

    monkeypatch.setattr(inference_module, "classify_object", fake_classify_object)

    def on_result(result, entry):
        events.results.append((result["id"], result["status"]))
        if entry.get("raise_result"):
            raise RuntimeError("result callback failed")

    def on_prediction(entry_id, response):
        events.predictions.append(entry_id)
        raise RuntimeError("prediction callback failed")

    results = batch_classify_objects(
        vlm=object(),
        entries=[
            {"id": "good", "images": ["front.png"], "text": "parallel"},
            {"id": "bad", "images": ["missing.png"], "text": "missing"},
            {
                "id": "boom",
                "images": ["front.png"],
                "text": "explode",
                "raise_result": True,
            },
        ],
        llm=object(),
        image_base_dir=tmp_path,
        on_progress=lambda entry_id, response: events.progress.append(entry_id),
        on_error=lambda entry_id, error: events.errors.append((entry_id, error)),
        on_result=on_result,
        on_prediction=on_prediction,
        max_workers=2,
        max_retries=1,
    )

    by_id = {result["id"]: result for result in results}
    assert by_id["good"]["status"] == "success"
    assert by_id["bad"]["status"] == "error"
    assert "Missing images" in by_id["bad"]["error"]
    assert by_id["boom"]["status"] == "error"
    assert by_id["boom"]["error"] == "classification exploded"
    assert events.progress == ["good"]
    assert {item[0] for item in events.errors} == {"bad", "boom"}
    assert events.predictions == ["good"]


def test_multi_prim_direct_parsing_strategies():
    answer_result = _parse_multi_prim_response(
        vlm_response='<answer>{"a": {"materials": {"material": "steel", "reason": "shiny"}}, "b": "plastic"}</answer>',
        object_ids=["a", "b"],
        output_key="classification",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert answer_result["a"]["classification"] == "steel"
    assert answer_result["a"]["reason"] == "shiny"
    assert answer_result["b"]["classification"] == "plastic"

    list_result = _parse_multi_prim_response(
        vlm_response='{"predictions": [{"object_id": "a", "material": "wood"}, {"path": "b", "materials": "glass"}]}',
        object_ids=["a", "b"],
        output_key="classification",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert list_result["a"]["classification"] == "wood"
    assert list_result["b"]["classification"] == "glass"

    embedded_result = _parse_multi_prim_response(
        vlm_response='preface {"a": {"class": "wheel"}} suffix',
        object_ids=["a"],
        output_key="class",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert embedded_result["a"]["class"] == "wheel"

    fallback_result = _parse_multi_prim_response(
        vlm_response="not json",
        object_ids=["a"],
        output_key="class",
        llm=_RetryingChatParser(
            content='<answer>{"a": {"class": "fallback"}}</answer>'
        ),
        system_prompt="system",
        text="text",
        max_retries=1,
        unknown_sentinel="unknown",
    )
    assert fallback_result["a"]["class"] == "fallback"

    assert (
        _parse_multi_prim_response(
            vlm_response="not json",
            object_ids=["a"],
            output_key="class",
            llm=_RetryingChatParser(content="still not json"),
            system_prompt="system",
            text="text",
            max_retries=1,
        )
        == {}
    )


def test_classify_objects_multi_prim_retries_and_pairs(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    tracker = TokenTracker()
    usage = TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5)

    with pytest.raises(ValueError, match="empty images"):
        classify_objects_multi_prim(
            vlm=_SequenceVLM([]),
            object_ids=["a"],
            text="text",
            images=[],
            llm=object(),
        )

    pair_vlm = _SequenceVLM(
        ['<answer>{"a": {"class": "chair"}}</answer>'],
        token_usage=usage,
    )
    result = classify_objects_multi_prim(
        vlm=pair_vlm,
        object_ids=["a"],
        text="text",
        images=["front.png"],
        llm=object(),
        invoke_kwargs={"temperature": 0.3, "max_tokens": 6},
        image_prompts=["caption"],
        token_tracker=tracker,
        max_retries=1,
    )
    assert result["a"]["class"] == "chair"
    assert pair_vlm.pair_calls[0]["max_tokens"] == 6
    assert tracker.get_stats()["invocation_count"] == 1

    retry_vlm = _SequenceVLM(["", '{"a": {"class": "after-empty"}}'])
    retry_result = classify_objects_multi_prim(
        vlm=retry_vlm,
        object_ids=["a"],
        text="text",
        images=["front.png"],
        llm=object(),
        invoke_kwargs={"max_completion_tokens": 4},
        image_prompts=["caption", "extra"],
        max_retries=2,
    )
    assert retry_result["a"]["class"] == "after-empty"
    assert [call["max_tokens"] for call in retry_vlm.calls] == [4, 8]

    error_retry_vlm = _SequenceVLM(
        [RuntimeError("temporary"), '{"a": {"class": "after-error"}}']
    )
    assert (
        classify_objects_multi_prim(
            vlm=error_retry_vlm,
            object_ids=["a"],
            text="text",
            images=["front.png"],
            llm=object(),
            max_retries=2,
        )["a"]["class"]
        == "after-error"
    )


@pytest.mark.asyncio
async def test_async_classify_object_retry_and_fallback_branches(monkeypatch):
    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    tracker = TokenTracker()
    usage = TokenUsage(input_tokens=3, output_tokens=4, total_tokens=7)
    vlm = _SequenceVLM(["", '{"class": "async-chair"}'], token_usage=usage)

    result = await async_classify_object(
        vlm=vlm,
        text="classify it",
        images=["front.png"],
        llm=object(),
        invoke_kwargs={"temperature": 0.2, "max_tokens": 4},
        image_prompts=["caption"],
        token_tracker=tracker,
        max_retries=2,
    )
    assert result["class"] == "async-chair"
    assert [call["max_tokens"] for call in vlm.async_pair_calls] == [4, 8]
    assert tracker.get_stats()["invocation_count"] == 2

    with pytest.raises(ValueError, match="empty or None images"):
        await async_classify_object(
            vlm=_SequenceVLM([]),
            text="classify it",
            images=[],
            llm=object(),
        )

    unable = await async_classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_RetryingChatParser(content="not json"),
        image_prompts=["one", "two"],
        max_retries=1,
    )
    assert unable == {
        "class": "Unable to parse",
        "original_response": "unstructured and undecidable",
    }

    class _AsyncFailingParser:
        async def ainvoke(self, *args, **kwargs):
            raise RuntimeError("async parser down")

    failed = await async_classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_AsyncFailingParser(),
        max_retries=1,
    )
    assert failed == {
        "class": "Error during parsing",
        "original_response": "unstructured and undecidable",
    }


@pytest.mark.asyncio
async def test_async_batch_classify_objects_branches(monkeypatch, tmp_path):
    image_path = tmp_path / "front.png"
    image_path.write_bytes(b"not really an image")
    events = SimpleNamespace(progress=[], errors=[], results=[], predictions=[])

    async def fake_async_classify_object(**kwargs):
        if "explode" in kwargs["text"]:
            raise RuntimeError("async classification exploded")
        return {"class": f"async-{kwargs['text']}"}

    monkeypatch.setattr(
        inference_module, "async_classify_object", fake_async_classify_object
    )

    def on_result(result, entry):
        events.results.append((result["id"], result["status"]))
        if entry.get("raise_result"):
            raise RuntimeError("result callback failed")

    def on_prediction(entry_id, response):
        events.predictions.append(entry_id)
        raise RuntimeError("prediction callback failed")

    results = await inference_module.async_batch_classify_objects(
        vlm=object(),
        entries=[
            {"id": "skip", "images": ["front.png"], "text": "skip"},
            {"id": "invalid", "images": [object()], "text": "invalid"},
            {
                "id": "good",
                "images": ["front.png"],
                "text": "ok",
                "image_metadata": [{"vlm_prompt": "caption"}],
            },
            {
                "id": "boom",
                "images": ["front.png"],
                "text": "explode",
                "raise_result": True,
            },
        ],
        llm=object(),
        image_base_dir=tmp_path,
        processed_ids={"skip"},
        on_progress=lambda entry_id, response: events.progress.append(entry_id),
        on_error=lambda entry_id, error: events.errors.append((entry_id, error)),
        on_result=on_result,
        on_prediction=on_prediction,
        max_workers=2,
        max_retries=1,
    )

    by_id = {result["id"]: result for result in results}
    assert by_id["invalid"]["status"] == "error"
    assert "Unsupported image types" in by_id["invalid"]["error"]
    assert by_id["good"]["vlm_response"]["class"] == "async-ok"
    assert by_id["boom"]["error"] == "async classification exploded"
    assert events.progress == ["good"]
    assert {item[0] for item in events.errors} == {"invalid", "boom"}
    assert events.predictions == ["good"]


class _SyncVLMParser:
    last_token_usage = None

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_remaining_single_object_branches(monkeypatch):
    assert (
        asyncio.run(
            _invoke_parser_model_async(
                _SyncVLMParser({"class": "dict-response"}),
                messages=[],
                parsing_prompt="parse me",
                parser_system_prompt="system",
                max_tokens=12,
            )
        )
        == "{'class': 'dict-response'}"
    )

    with pytest.raises(ValueError, match="empty or None images"):
        classify_object(
            vlm=_SequenceVLM([]),
            text="classify it",
            images=[],
            llm=object(),
        )

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    assert (
        classify_object(
            vlm=_SequenceVLM(["", '{"class": "after-empty"}']),
            text="classify it",
            images=["front.png"],
            llm=object(),
            max_retries=2,
        )["class"]
        == "after-empty"
    )

    final_empty = classify_object(
        vlm=_SequenceVLM([""]),
        text="classify it",
        images=["front.png"],
        llm=_RetryingChatParser(content="not json"),
        max_retries=1,
    )
    assert final_empty == {"class": "Unable to parse", "original_response": ""}

    empty_llm = classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_SequenceVLM([""]),
        max_retries=1,
    )
    assert empty_llm["class"] == "Unable to parse"


def test_direct_answer_json_fallback_branches(monkeypatch):
    monkeypatch.setattr(
        inference_module,
        "iter_json_dicts_from_llm_response",
        lambda response: iter(()),
    )

    material_answer = _parse_single_result_from_response_text(
        '<answer>{"material": "steel"}</answer>',
        output_key="classification",
    )
    assert material_answer == {"classification": "steel"}

    named_answer = _parse_single_result_from_response_text(
        '<answer>{"name": "bolt"}</answer>',
        output_key="classification",
    )
    assert named_answer == {"name": "bolt", "classification": "bolt"}

    monkeypatch.setattr(
        inference_module,
        "_extract_single_result_json_from_response_text",
        lambda response_text, *, output_key: None,
    )
    assert (
        classify_object(
            vlm=_SequenceVLM(['<answer>{"class": "direct"}</answer>']),
            text="classify it",
            images=["front.png"],
            llm=object(),
            max_retries=1,
        )["class"]
        == "direct"
    )

    assert (
        classify_object(
            vlm=_SequenceVLM(['<answer>{"material": "steel"}</answer>']),
            text="classify it",
            images=["front.png"],
            llm=object(),
            output_key="classification",
            max_retries=1,
        )["classification"]
        == "steel"
    )

    fallback_dict = classify_object(
        vlm=_SequenceVLM(['<answer>{"note": "not a class"}</answer>']),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=1,
    )
    assert fallback_dict["note"] == "not a class"

    fallback_list = classify_object(
        vlm=_SequenceVLM(["<answer>[]</answer>"]),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=1,
    )
    assert fallback_list["class"] == "[]"


def test_answer_block_direct_class_logs_without_material_extraction():
    result = classify_object(
        vlm=_SequenceVLM(['<answer>{"class": "wheel"}</answer>']),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=1,
    )
    assert result["class"] == "wheel"

    multiple_answers = classify_object(
        vlm=_SequenceVLM(
            ['<answer>{"class": "old"}</answer><answer>{"class": "new"}</answer>']
        ),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=1,
    )
    assert multiple_answers["class"] == "new"


def test_batch_duration_and_string_preview_branches(monkeypatch, tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"x")
    second.write_bytes(b"x")

    perf_values = iter([0.0, 3700.0, 3700.0, 3701.0])
    monkeypatch.setattr(inference_module, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        inference_module,
        "classify_object",
        lambda **kwargs: f"string-{Path(kwargs['images'][0]).name}",
    )

    results = batch_classify_objects(
        vlm=object(),
        entries=[
            {"id": "one", "images": [str(first)], "text": "one"},
            {"id": "two", "images": [str(second)], "text": "two"},
        ],
        llm=object(),
        max_retries=1,
    )

    assert [result["vlm_response"] for result in results] == [
        "string-first.png",
        "string-second.png",
    ]

    metadata = _extract_image_metadata_from_entry(
        {"media": {"images": [{"metadata": {"vlm_prompt": "caption"}}]}}
    )
    assert metadata == [{"vlm_prompt": "caption"}]


def test_parallel_duration_image_variants_and_string_preview(monkeypatch, tmp_path):
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"x")
    pil_image = PILImage.new("RGB", (1, 1))

    perf_values = iter([0.0, 3700.0, 3700.0, 3701.0])
    monkeypatch.setattr(inference_module, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        inference_module,
        "classify_object",
        lambda **kwargs: f"parallel-{kwargs['text']}",
    )

    results = batch_classify_objects(
        vlm=object(),
        entries=[
            {
                "id": "pil",
                "images": [pil_image],
                "text": "pil",
                "image_metadata": [{"vlm_prompt": "caption"}],
            },
            {
                "id": "path",
                "images": [str(existing)],
                "text": "path",
                "image_metadata": [{"missing_prompt": True}],
            },
            {"id": "invalid", "images": [object()], "text": "invalid"},
        ],
        llm=object(),
        on_error=lambda entry_id, error: None,
        max_workers=2,
        max_retries=1,
    )

    by_id = {result["id"]: result for result in results}
    assert by_id["pil"]["vlm_response"] == "parallel-pil"
    assert by_id["path"]["vlm_response"] == "parallel-path"
    assert by_id["invalid"]["status"] == "error"
    assert "Unsupported image types" in by_id["invalid"]["error"]


def test_multi_prim_remaining_parse_branches(monkeypatch):
    direct_parse = _parse_multi_prim_response(
        vlm_response='<answer>{"a": {"class": "direct-json"}}</answer>',
        object_ids=["a"],
        output_key="class",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert direct_parse["a"]["class"] == "direct-json"

    monkeypatch.setattr(
        inference_module,
        "extract_json_from_llm_response",
        lambda response: None,
    )

    direct_json = _parse_multi_prim_response(
        vlm_response='<answer>{"a": {"class": "json-loads"}}</answer>',
        object_ids=["a"],
        output_key="class",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert direct_json["a"]["class"] == "json-loads"

    embedded = _parse_multi_prim_response(
        vlm_response='prefix {"a": {"class": "embedded"}} suffix',
        object_ids=["a"],
        output_key="class",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert embedded["a"]["class"] == "embedded"

    invalid_answer = _parse_multi_prim_response(
        vlm_response="<answer>not json</answer>",
        object_ids=["a"],
        output_key="class",
        llm=_RetryingChatParser(content="not json either"),
        system_prompt="system",
        text="text",
        max_retries=1,
    )
    assert invalid_answer == {}


def test_multi_prim_payload_coercion_and_fallback_retry_branches(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    result = _parse_multi_prim_response(
        vlm_response=json.dumps(
            {
                "a": 3,
                "b": {"materials": {"classification": "steel"}},
                "c": {"result": {"material": "stone"}, "reason": "nested"},
                "d": {"class": ""},
                "items": [42],
            }
        ),
        object_ids=["a", "b", "c", "d"],
        output_key="classification",
        llm=object(),
        system_prompt="system",
        text="text",
        max_retries=1,
    )

    assert "a" not in result
    assert result["b"]["classification"] == "steel"
    assert result["c"]["classification"] == "stone"
    assert result["c"]["reason"] == "nested"
    assert "d" not in result

    empty_then_valid = _parse_multi_prim_response(
        vlm_response="not json",
        object_ids=["a"],
        output_key="class",
        llm=_SequenceVLM(["", '{"a": {"class": "after-empty"}}']),
        system_prompt="system",
        text="text",
        max_retries=2,
    )
    assert empty_then_valid["a"]["class"] == "after-empty"

    parsed_no_results_then_valid = _parse_multi_prim_response(
        vlm_response="not json",
        object_ids=["a"],
        output_key="class",
        llm=_SequenceVLM(['{"z": {"class": "wrong"}}', '{"a": {"class": "right"}}']),
        system_prompt="system",
        text="text",
        max_retries=2,
    )
    assert parsed_no_results_then_valid["a"]["class"] == "right"

    monkeypatch.setattr(
        inference_module,
        "extract_json_from_llm_response",
        lambda response: None,
    )
    assert (
        _parse_multi_prim_response(
            vlm_response="prefix {bad} suffix",
            object_ids=["a"],
            output_key="class",
            llm=_RetryingChatParser(content="not json"),
            system_prompt="system",
            text="text",
            max_retries=1,
        )
        == {}
    )

    assert (
        _parse_multi_prim_response(
            vlm_response="not json",
            object_ids=["a"],
            output_key="class",
            llm=_SequenceVLM([""]),
            system_prompt="system",
            text="text",
            max_retries=1,
        )
        == {}
    )


def test_classify_objects_multi_prim_empty_retry_without_token_budget(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    retry_vlm = _SequenceVLM(["", '{"a": {"class": "after-empty"}}'])

    result = classify_objects_multi_prim(
        vlm=retry_vlm,
        object_ids=["a"],
        text="text",
        images=["front.png"],
        llm=object(),
        max_retries=2,
    )
    assert result["a"]["class"] == "after-empty"

    final_empty = classify_objects_multi_prim(
        vlm=_SequenceVLM([""]),
        object_ids=["a"],
        text="text",
        images=["front.png"],
        llm=_RetryingChatParser(content="not json"),
        max_retries=1,
    )
    assert final_empty == {}


@pytest.mark.asyncio
async def test_async_classify_object_remaining_retry_and_sentinel_branches(monkeypatch):
    async def _no_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    result = await async_classify_object(
        vlm=_SequenceVLM(["", '{"class": "after-empty"}']),
        text="classify it",
        images=["front.png"],
        llm=object(),
        max_retries=2,
    )
    assert result["class"] == "after-empty"

    final_empty = await async_classify_object(
        vlm=_SequenceVLM([""]),
        text="classify it",
        images=["front.png"],
        llm=_RetryingChatParser(content="not json"),
        max_retries=1,
    )
    assert final_empty == {"class": "Unable to parse", "original_response": ""}

    sentinel_prompt = await async_classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_RetryingChatParser(content='{"class": "unknown"}'),
        max_retries=1,
        unknown_sentinel="unknown",
    )
    assert sentinel_prompt["class"] == "unknown"

    empty_then_parsed = await async_classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_SequenceVLM(["", '{"class": "fallback"}']),
        max_retries=2,
    )
    assert empty_then_parsed["class"] == "fallback"

    empty_llm = await async_classify_object(
        vlm=_SequenceVLM(["unstructured and undecidable"]),
        text="classify it",
        images=["front.png"],
        llm=_SequenceVLM([""]),
        max_retries=1,
    )
    assert empty_llm["class"] == "Unable to parse"


@pytest.mark.asyncio
async def test_async_batch_duration_image_variants_and_string_preview(
    monkeypatch, tmp_path
):
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"x")
    pil_image = PILImage.new("RGB", (1, 1))

    perf_values = iter([0.0, 3700.0, 3700.0, 3701.0])
    monkeypatch.setattr(inference_module, "perf_counter", lambda: next(perf_values))

    async def fake_async_classify_object(**kwargs):
        return f"async-{kwargs['text']}"

    monkeypatch.setattr(
        inference_module, "async_classify_object", fake_async_classify_object
    )

    results = await inference_module.async_batch_classify_objects(
        vlm=object(),
        entries=[
            {
                "id": "pil",
                "images": [pil_image],
                "text": "pil",
                "image_metadata": [{"vlm_prompt": "caption"}],
            },
            {
                "id": "path",
                "images": [str(existing)],
                "text": "path",
                "image_metadata": [{"missing_prompt": True}],
            },
            {"id": "missing", "images": ["missing.png"], "text": "missing"},
        ],
        llm=object(),
        max_workers=2,
        max_retries=1,
    )

    by_id = {result["id"]: result for result in results}
    assert by_id["pil"]["vlm_response"] == "async-pil"
    assert by_id["path"]["vlm_response"] == "async-path"
    assert by_id["missing"]["status"] == "error"
    assert "Missing images" in by_id["missing"]["error"]
