# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic object classification using Vision-Language Models.

This module provides the core classification logic extracted from material_agent,
made generic to work with any class labels.

Key functions:
- classify_object(): Classify a single object using VLM
- batch_classify_objects(): Batch classification with parallel/sequential processing
- async_classify_object(): Async version of classify_object
- async_batch_classify_objects(): Async batch classification using asyncio.gather
"""

import asyncio
import json
import logging
import os
import re
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Lock
from time import perf_counter
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from PIL import Image as PILImage

from world_understanding.functions.models.vision_language_models import (
    BaseVisionLanguageModel,
)
from world_understanding.utils.llm_parsing import (
    extract_json_from_llm_response,
    extract_last_answer_block,
    extract_material_from_json,
    iter_json_dicts_from_llm_response,
)
from world_understanding.utils.model_auth import (
    ModelAuthenticationFailure,
    raise_for_model_authentication,
)
from world_understanding.utils.token_tracking import TokenTracker

logger = logging.getLogger(__name__)

_DEFAULT_VLM_GENERATE_TIMEOUT_SECONDS = 180.0
_UNSTRUCTURED_LABEL_PATTERNS = (
    re.compile(
        r"\b(?:must|should|would|can|will|is|are)?\s*(?:be\s+)?"
        r"classified\s+as\s+[\"'`]?"
        r"(?P<value>[A-Za-z0-9][A-Za-z0-9 _/\-]{0,80})[\"'`]?",
        re.IGNORECASE,
    ),
    re.compile(
        # Avoid bare "class is" because it also appears in incidental prose.
        r"\b(?:classification|label|answer|(?:final|selected|chosen|predicted|"
        r"object|component)\s+class)\s*(?:is|:|=)\s*[\"'`]?"
        r"(?P<value>[A-Za-z0-9][A-Za-z0-9 _/\-]{0,80})[\"'`]?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:falls?|fall)\s+under\s+(?:the\s+)?[\"'`]?"
        r"(?P<value>[A-Za-z0-9][A-Za-z0-9 _/\-]{0,80}?)[\"'`]?"
        r"\s+(?:category|class|classification|label)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:belongs?|belong)\s+to\s+(?:the\s+)?[\"'`]?"
        r"(?P<value>[A-Za-z0-9][A-Za-z0-9 _/\-]{0,80}?)[\"'`]?"
        r"\s+(?:category|class|classification|label)\b",
        re.IGNORECASE,
    ),
)
_NEGATED_LABEL_CONTEXT_RE = re.compile(
    r"\b(?:not|never|no|cannot|can't|couldn't|shouldn't|won't|isn't|aren't|"
    r"mustn't|must\s+not|should\s+not|could\s+not|would\s+not)\b",
    re.IGNORECASE,
)


def extract_answer_block(text: str) -> str | None:
    """Extract content from the LAST <answer></answer> block in text.

    This uses the last answer block because VLMs often repeat the example
    from the prompt (which may contain an answer block) before providing
    their actual answer.

    Args:
        text: Text potentially containing <answer> tags

    Returns:
        Content within the last answer block, or None if not found
    """
    return extract_last_answer_block(text)


def get_fibonacci_delay(attempt: int, base_delay: float = 1.0) -> float:
    """Calculate delay using Fibonacci sequence.

    Args:
        attempt: The current attempt number (0-indexed)
        base_delay: Base delay multiplier (default: 1.0 seconds)

    Returns:
        Delay in seconds following Fibonacci pattern

    Examples:
        attempt 0 → 1 * base_delay
        attempt 1 → 1 * base_delay
        attempt 2 → 2 * base_delay
        attempt 3 → 3 * base_delay
        attempt 4 → 5 * base_delay
    """
    if attempt <= 0:
        return base_delay
    elif attempt == 1:
        return base_delay

    # Generate Fibonacci number for the attempt
    fib = [1, 1]
    for _ in range(2, attempt + 1):
        fib.append(fib[-1] + fib[-2])

    return fib[attempt] * base_delay


def _get_vlm_generate_timeout_seconds() -> float:
    """Return the hard deadline for VLM generation calls."""
    raw_timeout = os.environ.get(
        "WU_VLM_GENERATE_TIMEOUT_SECONDS",
        str(_DEFAULT_VLM_GENERATE_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid WU_VLM_GENERATE_TIMEOUT_SECONDS=%r; falling back to %.0fs",
            raw_timeout,
            _DEFAULT_VLM_GENERATE_TIMEOUT_SECONDS,
        )
        timeout_seconds = _DEFAULT_VLM_GENERATE_TIMEOUT_SECONDS

    if timeout_seconds > 0:
        return timeout_seconds
    return _DEFAULT_VLM_GENERATE_TIMEOUT_SECONDS


def _call_sync_with_timeout(
    func: Any,
    *,
    timeout_seconds: float,
    operation_name: str,
) -> Any:
    """Execute a synchronous callable with a hard deadline."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"{operation_name} did not respond within {timeout_seconds:.0f}s"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


async def _call_async_with_timeout(
    awaitable: Any,
    *,
    timeout_seconds: float,
    operation_name: str,
) -> Any:
    """Await a coroutine with a hard deadline."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except TimeoutError as exc:
        raise TimeoutError(
            f"{operation_name} did not respond within {timeout_seconds:.0f}s"
        ) from exc


def _invoke_parser_with_chat_model(
    parser_model: Any,
    messages: list[Any],
    *,
    max_tokens: int,
) -> str:
    """Invoke a chat-style parser model and return plain text content."""
    if (
        hasattr(parser_model, "model")
        and "gpt-5" in str(getattr(parser_model, "model", "")).lower()
    ):
        try:
            response = parser_model.invoke(messages, max_completion_tokens=max_tokens)
        except Exception as e:
            raise_for_model_authentication(e)
            raise
    else:
        try:
            response = parser_model.invoke(
                messages, temperature=0.1, max_tokens=max_tokens
            )
        except Exception as e:
            raise_for_model_authentication(e)
            error_msg = str(e)
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                try:
                    response = parser_model.invoke(
                        messages, max_completion_tokens=max_tokens
                    )
                except Exception as retry_error:
                    raise_for_model_authentication(retry_error)
                    raise
            else:
                raise

    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


async def _ainvoke_parser_with_chat_model(
    parser_model: Any,
    messages: list[Any],
    *,
    max_tokens: int,
) -> str:
    """Invoke a chat-style parser model asynchronously and return text content."""
    if hasattr(parser_model, "ainvoke"):
        try:
            response = await parser_model.ainvoke(
                messages, temperature=0.1, max_tokens=max_tokens
            )
        except Exception as e:
            raise_for_model_authentication(e)
            error_msg = str(e)
            if "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
                try:
                    response = await parser_model.ainvoke(
                        messages, max_completion_tokens=max_tokens
                    )
                except Exception as retry_error:
                    raise_for_model_authentication(retry_error)
                    raise
            else:
                raise
    else:
        response = await asyncio.to_thread(
            _invoke_parser_with_chat_model,
            parser_model,
            messages,
            max_tokens=max_tokens,
        )

    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _invoke_parser_with_vlm(
    parser_model: Any,
    parsing_prompt: str,
    *,
    parser_system_prompt: str,
    max_tokens: int,
) -> str:
    """Invoke a VLM parser in text-only mode and return plain text content."""
    timeout_seconds = _get_vlm_generate_timeout_seconds()

    def _generate() -> str:
        return parser_model.generate(
            prompt=parsing_prompt,
            images=None,
            system_prompt=parser_system_prompt,
            temperature=0.1,
            max_tokens=max_tokens,
        )

    result = _call_sync_with_timeout(
        _generate,
        timeout_seconds=timeout_seconds,
        operation_name="Parser VLM generate",
    )
    return result if isinstance(result, str) else str(result)


async def _ainvoke_parser_with_vlm(
    parser_model: Any,
    parsing_prompt: str,
    *,
    parser_system_prompt: str,
    max_tokens: int,
) -> str:
    """Invoke a VLM parser in text-only mode asynchronously and return text."""
    timeout_seconds = _get_vlm_generate_timeout_seconds()

    if hasattr(parser_model, "agenerate"):
        result = await _call_async_with_timeout(
            parser_model.agenerate(
                prompt=parsing_prompt,
                images=None,
                system_prompt=parser_system_prompt,
                temperature=0.1,
                max_tokens=max_tokens,
            ),
            timeout_seconds=timeout_seconds,
            operation_name="Parser VLM agenerate",
        )
    else:
        result = await asyncio.to_thread(
            _invoke_parser_with_vlm,
            parser_model,
            parsing_prompt,
            parser_system_prompt=parser_system_prompt,
            max_tokens=max_tokens,
        )

    return result if isinstance(result, str) else str(result)


def _invoke_parser_model_sync(
    parser_model: Any,
    *,
    messages: list[Any],
    parsing_prompt: str,
    parser_system_prompt: str,
    max_tokens: int,
) -> str:
    """Invoke either a chat parser model or a VLM parser model."""
    if hasattr(parser_model, "invoke"):
        return _invoke_parser_with_chat_model(
            parser_model,
            messages,
            max_tokens=max_tokens,
        )

    if isinstance(parser_model, BaseVisionLanguageModel) or hasattr(
        parser_model, "generate"
    ):
        return _invoke_parser_with_vlm(
            parser_model,
            parsing_prompt,
            parser_system_prompt=parser_system_prompt,
            max_tokens=max_tokens,
        )

    raise TypeError(
        "Parser model must support chat invoke/ainvoke or VLM generate/agenerate"
    )


async def _invoke_parser_model_async(
    parser_model: Any,
    *,
    messages: list[Any],
    parsing_prompt: str,
    parser_system_prompt: str,
    max_tokens: int,
) -> str:
    """Async wrapper for invoking either a chat parser model or a VLM parser."""
    if hasattr(parser_model, "ainvoke") or hasattr(parser_model, "invoke"):
        return await _ainvoke_parser_with_chat_model(
            parser_model,
            messages,
            max_tokens=max_tokens,
        )

    if (
        isinstance(parser_model, BaseVisionLanguageModel)
        or hasattr(parser_model, "agenerate")
        or hasattr(parser_model, "generate")
    ):
        return await _ainvoke_parser_with_vlm(
            parser_model,
            parsing_prompt,
            parser_system_prompt=parser_system_prompt,
            max_tokens=max_tokens,
        )

    raise TypeError(
        "Parser model must support chat invoke/ainvoke or VLM generate/agenerate"
    )


def _extract_single_result_json_from_response_text(
    response_text: str,
    *,
    output_key: str,
) -> dict[str, Any] | None:
    """Extract JSON compatible with the requested single-result output key."""
    legacy_result = None
    for candidate in iter_json_dicts_from_llm_response(response_text):
        if output_key in candidate:
            return candidate
        if extract_material_from_json(candidate):
            legacy_result = candidate
    return legacy_result


def _parse_single_result_from_response_text(
    response_text: str,
    *,
    output_key: str,
    unknown_sentinel: str | None = None,
) -> dict[str, Any] | None:
    """Parse a structured single-object classification result from text."""
    if not response_text or not response_text.strip():
        return None

    answer_content = extract_answer_block(response_text)
    result = None
    answer_fallback: dict[str, Any] | None = None

    if answer_content:
        all_answers = re.findall(
            r"<answer[^>]*>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE
        )
        if len(all_answers) > 1:
            logger.debug(
                "Parser returned %d answer blocks, using the last one",
                len(all_answers),
            )

        result = _extract_single_result_json_from_response_text(
            answer_content,
            output_key=output_key,
        )

        if result:
            value = extract_material_from_json(result)
            if value:
                _rename_legacy_material_key(
                    result,
                    output_key=output_key,
                    value=value,
                )
            return result

        try:
            parsed_content = json.loads(answer_content)
            if isinstance(parsed_content, dict):
                value = extract_material_from_json(parsed_content)
                if value or output_key in parsed_content:
                    result = dict(parsed_content)
                    if output_key != "material" and "material" in result:
                        _rename_legacy_material_key(
                            result,
                            output_key=output_key,
                            value=value,
                        )
                    elif output_key not in result:
                        result[output_key] = value
                    return result
                answer_fallback = {output_key: answer_content}
            else:
                answer_fallback = {output_key: answer_content}
        except json.JSONDecodeError:
            answer_fallback = {output_key: answer_content}

    result = _extract_single_result_json_from_response_text(
        response_text,
        output_key=output_key,
    )
    if result:
        value = extract_material_from_json(result)
        if value:
            _rename_legacy_material_key(
                result,
                output_key=output_key,
                value=value,
            )
        return result

    result = _explicit_unstructured_label_result(
        response_text,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    if result:
        return result

    if answer_fallback:
        return answer_fallback

    return None


def _rename_legacy_material_key(
    result: dict[str, Any],
    *,
    output_key: str,
    value: Any,
) -> None:
    """Rename the legacy primary-label ``material`` key when needed.

    Some prompts emit the primary label under ``material`` even when the caller
    configured a different ``output_key`` such as ``classification``. Only treat
    ``material`` as the legacy primary label when ``output_key`` is missing; if
    the response already has ``output_key``, leave any sibling ``material`` field
    intact as structured metadata.
    """
    if output_key in result:
        return
    if output_key != "material" and "material" in result:
        result[output_key] = value
        del result["material"]
        return
    result[output_key] = value


def _extract_explicit_unstructured_label(response_text: str) -> str | None:
    """Extract a plainly stated class label from non-JSON model text."""
    if not response_text:
        return None

    matches: list[tuple[int, str]] = []
    for pattern in _UNSTRUCTURED_LABEL_PATTERNS:
        for match in pattern.finditer(response_text):
            if _label_match_is_negated(response_text, match.start()):
                continue
            label = _clean_unstructured_label(match.group("value"))
            if label:
                matches.append((match.start(), label))

    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _label_match_is_negated(response_text: str, match_start: int) -> bool:
    # Keep the window local so unrelated earlier negations do not suppress
    # a later affirmative label.
    local_context = response_text[max(0, match_start - 48) : match_start]
    return bool(_NEGATED_LABEL_CONTEXT_RE.search(local_context))


def _clean_unstructured_label(value: str) -> str | None:
    label = value.strip().strip("\"'`")
    # Preserve conjunctions/prepositions inside labels such as "black and white"
    # or "adapter for rail".
    label = re.split(
        r"\b(?:because|since|when|while|but)\b",
        label,
        maxsplit=1,
    )[0]
    label = label.strip().strip(" .,:;\"'`")

    if not label or len(label) > 80:
        return None
    if not re.search(r"[A-Za-z0-9]", label):
        return None
    return label


def _explicit_unknown_sentinel_result(
    response_text: str,
    *,
    output_key: str,
    unknown_sentinel: str | None,
) -> dict[str, str] | None:
    """Return a sentinel result when the raw VLM response is exactly it."""
    if not unknown_sentinel or not response_text:
        return None

    answer_content = extract_answer_block(response_text)
    if answer_content and _is_exact_unknown_sentinel_text(
        answer_content, unknown_sentinel
    ):
        return {output_key: unknown_sentinel, "original_response": response_text}

    if not _is_exact_unknown_sentinel_text(response_text, unknown_sentinel):
        return None
    return {output_key: unknown_sentinel, "original_response": response_text}


def _is_exact_unknown_sentinel_text(text: str, unknown_sentinel: str | None) -> bool:
    """Return True when text is only the configured sentinel value."""
    if not unknown_sentinel:
        return False

    candidate = text.strip()
    sentinel = unknown_sentinel.strip()
    if not sentinel:
        return False
    if candidate.lower() == sentinel.lower():
        return True
    if _is_wrapped_literal(sentinel):
        return False

    unwrapped_candidate = _unwrap_transport_literal(candidate)
    return unwrapped_candidate.lower() == sentinel.lower()


def _is_wrapped_literal(text: str) -> bool:
    """Return True when text is intentionally wrapped in matching quote marks."""
    return len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`"


def _unwrap_transport_literal(text: str) -> str:
    """Remove only transport quotes around a model-returned scalar value."""
    if _is_wrapped_literal(text):
        return text[1:-1].strip()
    return text


def _normalize_unknown_sentinel_result(
    result: dict[str, Any],
    *,
    output_key: str,
    unknown_sentinel: str | None,
) -> None:
    """Canonicalize result values that are exactly the configured sentinel."""
    if not unknown_sentinel:
        return
    value = result.get(output_key)
    if isinstance(value, str) and _is_exact_unknown_sentinel_text(
        value, unknown_sentinel
    ):
        result[output_key] = unknown_sentinel


def _explicit_unstructured_label_result(
    response_text: str,
    *,
    output_key: str,
    unknown_sentinel: str | None,
) -> dict[str, Any] | None:
    """Build a normalized result from a plainly stated non-JSON label."""
    unstructured_label = _extract_explicit_unstructured_label(response_text)
    if not unstructured_label:
        return None

    result: dict[str, Any] = {output_key: unstructured_label}
    _normalize_unknown_sentinel_result(
        result,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    return result


def classify_object(
    vlm: BaseVisionLanguageModel,
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    output_key: str = "class",
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> dict[str, Any]:
    """Classify an object using Vision-Language Model.

    This is the generic classification function that can work with any class labels.
    It's the core function extracted from material_agent's assign_material().

    Args:
        vlm: Vision-Language Model instance to use for inference
        text: Context text containing object description and available classes
        images: List of images as file paths (str/Path) or PIL Image objects
        llm: LLM for parsing VLM response into structured format (fallback only)
        system_prompt: Optional custom system prompt (uses default if None)
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        image_prompts: Optional list of prompts/captions for each image
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)
        output_key: Key name for the classification result in output dict (default: "class")
                   Examples: "class", "material", "vehicle_type", "pattern"
        token_tracker: Optional TokenTracker to collect usage statistics
        unknown_sentinel: Optional explicit sentinel value to preserve when the
            VLM says the object is unknown/unclassifiable

    Returns:
        Dict with output_key and "original_response" keys

    Example:
        ```python
        from PIL import Image
        from world_understanding.functions.models.vision_language_models import create_vlm
        from world_understanding.functions.models.chat_models import create_chat_model

        # Create VLM and LLM
        vlm = create_vlm(backend="nim", model="google/gemma-4-31b-it")
        llm = create_chat_model(backend="nim")

        # Classify vehicle type
        text = "This is a vehicle. Available types: sedan, SUV, truck, van."
        images = ["car_front.png", "car_side.png"]

        response = classify_object(vlm, text, images, llm, output_key="vehicle_type")
        print(response)
        # Output: {
        #     "vehicle_type": "sedan",
        #     "original_response": "Based on the images, this is a sedan..."
        # }

        # Classify material with custom params
        text = "This is a car wheel. Materials: steel, rubber, plastic."
        invoke_kwargs = {"temperature": 0.5, "max_tokens": 512}
        response = classify_object(vlm, text, images, llm,
                                   output_key="material",
                                   invoke_kwargs=invoke_kwargs)
        # Output: {"material": "rubber", "original_response": "..."}
        ```
    """
    # Extract temperature and max_tokens from invoke_kwargs if present
    temperature = None
    max_tokens = None
    if invoke_kwargs:
        temperature = invoke_kwargs.get("temperature")
        mt = invoke_kwargs.get("max_tokens")
        max_tokens = (
            mt if mt is not None else invoke_kwargs.get("max_completion_tokens")
        )
    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = (
            "You are an expert at identifying objects and their properties. "
            "Analyze the images carefully and provide clear reasoning for your "
            "classification."
        )

    # Use the text prompt as-is (respect user's instructions)
    prompt = text

    # Check for empty or None images
    if not images or len(images) == 0:
        error_msg = (
            f"classify_object called with empty or None images list! "
            f"images={images}. The VLM will not be able to analyze anything."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Running classification with {len(images)} images")
    logger.debug(f"Images parameter type: {type(images)}, value: {images}")
    logger.debug(f"System prompt: {system_prompt[:100]}...")
    logger.debug(f"Prompt: {prompt[:100]}...")  # Log first 100 chars
    logger.debug(f"Output key: {output_key}")

    # Run VLM inference with retry logic for empty responses
    vlm_response = ""
    current_max_tokens = max_tokens  # Track current token limit for adaptive retry
    timeout_seconds = _get_vlm_generate_timeout_seconds()

    for attempt in range(max_retries):
        try:
            if image_prompts and len(image_prompts) == len(images):
                # Use image-caption pairs when prompts are provided
                logger.debug("Using image-caption pairs for VLM inference")
                image_caption_pairs = list(zip(image_prompts, images, strict=False))

                def _generate_with_pairs(
                    image_caption_pairs=image_caption_pairs,
                    current_max_tokens=current_max_tokens,
                ) -> str:
                    return vlm.generate_with_image_caption_pairs(
                        image_caption_pairs=image_caption_pairs,
                        final_prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                    )

                vlm_response = _call_sync_with_timeout(
                    _generate_with_pairs,
                    timeout_seconds=timeout_seconds,
                    operation_name="VLM generate_with_image_caption_pairs",
                )
            else:
                # Standard generation without individual image captions
                if image_prompts and len(image_prompts) != len(images):
                    logger.warning(
                        f"Image prompts count ({len(image_prompts)}) doesn't match "
                        f"images count ({len(images)}), ignoring prompts"
                    )

                def _generate(
                    current_max_tokens=current_max_tokens,
                ) -> str:
                    return vlm.generate(
                        prompt=prompt,
                        images=images,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                    )

                vlm_response = _call_sync_with_timeout(
                    _generate,
                    timeout_seconds=timeout_seconds,
                    operation_name="VLM generate",
                )

            # Track token usage if tracker provided
            if token_tracker is not None and vlm.last_token_usage is not None:
                token_tracker.add_usage(vlm.last_token_usage)

            # Check if response is empty or just whitespace
            if vlm_response and vlm_response.strip():
                # Got a valid response, break out of retry loop
                logger.debug(
                    f"VLM response received on attempt {attempt + 1}/{max_retries}"
                )
                break
            else:
                # Empty response, retry if we have attempts left
                if attempt < max_retries - 1:
                    # Double max_tokens for next attempt (helps with reasoning models like GPT-5)
                    if current_max_tokens:
                        previous_tokens = current_max_tokens
                        current_max_tokens = current_max_tokens * 2
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}. "
                            f"This may be due to max_completion_tokens budget being exceeded (common with high reasoning_effort). "
                            f"Doubling max_tokens from {previous_tokens} to {current_max_tokens} for next attempt."
                        )
                    else:
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}, retrying..."
                        )

                    retry_delay = get_fibonacci_delay(attempt, base_delay=1.0)
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    import time

                    time.sleep(retry_delay)
                else:
                    logger.error(f"Empty VLM response after {max_retries} attempts")
                    vlm_response = ""

        except Exception as e:
            raise_for_model_authentication(e)
            logger.error(
                f"VLM inference error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt < max_retries - 1:
                retry_delay = get_fibonacci_delay(attempt, base_delay=1.0)
                logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                import time

                time.sleep(retry_delay)
            else:
                raise

    logger.debug(
        f"VLM response: {vlm_response[:100] if vlm_response else 'Empty'}..."
    )  # Log first 100 chars

    # First, check if there's an <answer> block (use the last one if multiple exist)
    logger.debug("Looking for <answer> block(s) in VLM response")
    answer_content = extract_answer_block(vlm_response)

    result = None
    answer_fallback: dict[str, Any] | None = None

    if answer_content:
        # Check if there were multiple answer blocks
        all_answers = re.findall(
            r"<answer[^>]*>(.*?)</answer>", vlm_response, re.DOTALL | re.IGNORECASE
        )
        if len(all_answers) > 1:
            logger.debug(f"Found {len(all_answers)} answer blocks, using the last one")
        logger.debug(f"Using answer block content: {answer_content[:100]}...")

        # Try to extract JSON from the answer block
        result = _extract_single_result_json_from_response_text(
            answer_content,
            output_key=output_key,
        )

        if result:
            # Use robust extraction to handle various schemas
            # Note: extract_material_from_json is generic despite its name
            value = extract_material_from_json(result)
            if value:
                # Preserve all fields from original JSON, but ensure output_key is set
                _rename_legacy_material_key(
                    result,
                    output_key=output_key,
                    value=value,
                )
                logger.info(
                    f"Successfully extracted JSON from answer block: {output_key}='{value}'"
                )
            else:
                logger.info(
                    f"Successfully extracted JSON from answer block: {output_key}='{result.get(output_key)}'"
                )
        else:
            # If no valid JSON in answer block, try to parse the content as JSON directly
            try:
                result = json.loads(answer_content)
                if isinstance(result, dict):
                    # Use robust extraction for any valid dict structure
                    value = extract_material_from_json(result)
                    if value or output_key in result:
                        logger.info(
                            f"Successfully parsed answer block as JSON: {output_key}='{value or result.get(output_key)}'"
                        )
                        if value:
                            _rename_legacy_material_key(
                                result,
                                output_key=output_key,
                                value=value,
                            )
                    else:
                        # Not a valid material JSON, use the content as-is
                        logger.info(
                            f"Using answer block content as {output_key}: {answer_content}"
                        )
                        answer_fallback = {output_key: answer_content}
                else:
                    # Not a dict, use content as-is
                    logger.info(
                        f"Using answer block content as {output_key}: {answer_content}"
                    )
                    answer_fallback = {output_key: answer_content}
            except json.JSONDecodeError:
                # Use the answer content as the value directly
                logger.info(
                    f"Using answer block content as {output_key} (not JSON): {answer_content}"
                )
                answer_fallback = {output_key: answer_content}

        # Add the original response
        if result:
            _normalize_unknown_sentinel_result(
                result,
                output_key=output_key,
                unknown_sentinel=unknown_sentinel,
            )
            result["original_response"] = vlm_response
            return result

    # Fall back to extracting JSON from the entire response before accepting a
    # non-JSON answer block. Some models emit a valid JSON code block followed
    # by a stale prompt placeholder such as "<answer>your answer</answer>".
    logger.debug("Attempting to extract JSON from entire response")
    result = _extract_single_result_json_from_response_text(
        vlm_response,
        output_key=output_key,
    )

    if result:
        # Use robust extraction to handle various schemas
        value = extract_material_from_json(result)
        if value:
            # Preserve all fields from original JSON, but ensure output_key is set
            _rename_legacy_material_key(
                result,
                output_key=output_key,
                value=value,
            )
            logger.info(
                f"Successfully extracted JSON from VLM response: {output_key}='{value}'"
            )
        else:
            logger.info(
                f"Successfully extracted JSON from VLM response: {output_key}='{result.get(output_key)}'"
            )
        # Add the original VLM response to the result
        _normalize_unknown_sentinel_result(
            result,
            output_key=output_key,
            unknown_sentinel=unknown_sentinel,
        )
        result["original_response"] = vlm_response
        return result

    sentinel_result = _explicit_unknown_sentinel_result(
        vlm_response,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    if sentinel_result:
        logger.info(
            f"Preserving explicit unknown sentinel for {output_key}: "
            f"'{unknown_sentinel}'"
        )
        return sentinel_result

    result = _explicit_unstructured_label_result(
        vlm_response,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    if result:
        logger.info(
            "Extracted explicit unstructured classification from VLM response: %s='%s'",
            output_key,
            result.get(output_key),
        )
        result["original_response"] = vlm_response
        return result

    if answer_fallback:
        answer_fallback["original_response"] = vlm_response
        return answer_fallback

    # If direct extraction failed, use LLM as fallback
    logger.debug("Direct JSON extraction failed, using LLM to parse VLM response")

    # Create comprehensive prompt for LLM fallback with full context
    if unknown_sentinel:
        unknown_instruction = (
            f'\n- If the VLM response is exactly "{unknown_sentinel}", '
            f'preserve that value exactly as "{output_key}".'
        )
        prediction_instruction = (
            "- Preserve explicit configured sentinel values from the VLM response. "
            "If there is no visible evidence and the original prompt defines the "
            f'sentinel "{unknown_sentinel}", return that sentinel instead of guessing.'
        )
        value_instruction = (
            "The value must match exactly as it appears in the options list, or "
            f'must be "{unknown_sentinel}" when the response is explicitly unknown.'
        )
        parser_system_choice = (
            "Choose a real option from the available list unless the VLM explicitly "
            f'returned the configured sentinel "{unknown_sentinel}".'
        )
    else:
        unknown_instruction = ""
        prediction_instruction = (
            '- DO NOT return "unknown" - make an informed choice from the '
            "available options"
        )
        value_instruction = (
            "The value must match exactly as it appears in the options list. "
            'Always choose a real option from the list, never "unknown".'
        )
        parser_system_choice = (
            "Always choose a real option from the available list - be decisive "
            "and make the best choice based on the object context."
        )

    parsing_prompt = f"""CONTEXT: You are acting as an intelligent fallback because the VLM failed to return properly structured JSON output.

The VLM was asked to analyze images and select a class/label, but its response was not in the expected JSON format. You have two options:

1. **EXTRACT**: If the VLM response contains a clear selection, extract it
2. **PREDICT**: If the VLM response is unclear/invalid, use the same context to make your own prediction

ORIGINAL VLM SYSTEM PROMPT (with classes list):
{system_prompt}

ORIGINAL VLM USER PROMPT/CONTEXT (describes the object):
{text}

VLM'S ACTUAL RESPONSE (unstructured):
{vlm_response}

YOUR TASK:
**Step 1 - Try to Extract:**
- Look for <answer> blocks first, then <reasoning> blocks, then overall response
- If VLM mentioned multiple options, choose the most specific/final one
- The value MUST be from the available options list in the system prompt
{unknown_instruction}

**Step 2 - If Extract Fails, Make Your Own Prediction:**
- Use the VLM system prompt (classification expertise) + user context (object description)
- Analyze what class would be most appropriate for this object
- Consider the object type, function, typical classifications
- Select the BEST MATCH from the available options list
{prediction_instruction}

AVAILABLE OPTIONS LIST (extract from system prompt above):
Look for the options list in the VLM system prompt and use ONLY those options.

OUTPUT REQUIREMENTS:
Return ONLY a JSON object with this exact structure:
{{
    "{output_key}": "selected_value"
}}

{value_instruction}
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        parser_system_prompt = (
            "You are an intelligent LLM fallback for classification. A Vision-Language Model (VLM) "
            "failed to return structured output. You can either extract the value from the VLM response, "
            "OR if that's unclear, make your own informed prediction using the same context "
            f"(object description, available options) that the VLM had. {parser_system_choice}"
        )
        messages = [
            SystemMessage(content=parser_system_prompt),
            HumanMessage(content=parsing_prompt),
        ]

        parsed_response = ""

        for llm_attempt in range(max_retries):
            try:
                parsed_response = _invoke_parser_model_sync(
                    llm,
                    messages=messages,
                    parsing_prompt=parsing_prompt,
                    parser_system_prompt=parser_system_prompt,
                    max_tokens=256,
                )

                if parsed_response and parsed_response.strip():
                    logger.debug(
                        f"LLM parsing response received on attempt {llm_attempt + 1}/{max_retries}"
                    )
                    break
                else:
                    if llm_attempt < max_retries - 1:
                        llm_retry_delay = get_fibonacci_delay(
                            llm_attempt, base_delay=0.5
                        )
                        logger.warning(
                            f"Empty LLM parsing response on attempt {llm_attempt + 1}/{max_retries}, retrying in {llm_retry_delay:.1f} seconds..."
                        )
                        import time

                        time.sleep(llm_retry_delay)
                    else:
                        logger.error(
                            f"Empty LLM parsing response after {max_retries} attempts"
                        )
                        parsed_response = ""

            except Exception as e:
                raise_for_model_authentication(e)
                logger.error(
                    f"LLM parsing error on attempt {llm_attempt + 1}/{max_retries}: {e}"
                )
                if llm_attempt < max_retries - 1:
                    llm_retry_delay = get_fibonacci_delay(llm_attempt, base_delay=0.5)
                    logger.info(
                        f"Retrying LLM parsing in {llm_retry_delay:.1f} seconds..."
                    )
                    import time

                    time.sleep(llm_retry_delay)
                else:
                    raise

        result = _parse_single_result_from_response_text(
            parsed_response,
            output_key=output_key,
            unknown_sentinel=unknown_sentinel,
        )

        if result:
            logger.info(
                f"Successfully parsed to structured format using LLM: {output_key}='{result.get(output_key)}'"
            )
            # Add the original VLM response to the result
            _normalize_unknown_sentinel_result(
                result,
                output_key=output_key,
                unknown_sentinel=unknown_sentinel,
            )
            result["original_response"] = vlm_response
            return result
        else:
            # Fallback: return a dict with the raw response
            logger.warning(
                "Failed to parse LLM response to JSON, returning fallback structure"
            )
            return {
                output_key: "Unable to parse",
                "original_response": vlm_response,
            }

    except Exception as e:
        raise_for_model_authentication(e)
        logger.error(f"Error parsing VLM response with LLM: {e}")
        # Fallback: return a dict with the raw response
        return {
            output_key: "Error during parsing",
            "original_response": vlm_response,
        }


def batch_classify_objects(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None = None,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    on_progress: Any | None = None,
    on_error: Any | None = None,
    processed_ids: set[str] | None = None,
    on_result: Any | None = None,
    on_prediction: Any | None = None,
    max_workers: int | None = None,
    max_retries: int = 3,
    output_key: str = "class",
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> list[dict[str, Any]]:
    """Process multiple classification tasks in batch with optional parallel execution.

    Args:
        vlm: Vision-Language Model instance to use for inference
        entries: List of dictionaries containing:
            - id: Unique identifier
            - text: Context text with object and available classes
            - images: List of images (str/Path filenames or PIL Image objects)
        llm: LLM for parsing VLM responses into structured format
        image_base_dir: Base directory for image files (if not absolute paths)
                       Note: Only applies to string/Path images, not PIL Images
        system_prompt: Optional custom system prompt
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        on_progress: Optional callback function(entry_id, response) called after each item
        on_error: Optional callback function(entry_id, error) called on errors
        processed_ids: Set of entry IDs to skip (for resuming interrupted runs)
        on_result: Optional callback function(result, entry) called after each item
        on_prediction: Optional callback function(entry_id, classification_dict) called for each successful prediction
        max_workers: Maximum number of parallel workers (default: None = sequential, 1 = sequential,
                    >1 = parallel). Uses ThreadPoolExecutor for concurrent VLM API calls.
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)
        output_key: Key name for the classification result (default: "class")
        token_tracker: Optional TokenTracker to collect usage statistics across all calls
        unknown_sentinel: Optional explicit sentinel value to preserve when the
            VLM says an object is unknown/unclassifiable

    Returns:
        List of dictionaries containing:
            - id: Entry identifier
            - vlm_response: Classification response (dict with output_key and "original_response")
            - status: "success" or "error"
            - error: Error message (if status is "error")

    Example:
        ```python
        from PIL import Image

        # Example with parallel processing
        entries = [
            {
                "id": "vehicle_001",
                "text": "This is a vehicle. Types: sedan, SUV, truck.",
                "images": ["vehicle1.png", "vehicle2.png"]  # File paths
            },
            {
                "id": "vehicle_002",
                "text": "This is a vehicle. Types: sedan, SUV, truck.",
                "images": [Image.open("door1.png"), Image.open("door2.png")]  # PIL Images
            }
        ]

        # Sequential processing (default)
        results = batch_classify_objects(vlm=vlm, entries=entries, llm=llm, output_key="vehicle_type")

        # Parallel processing with 4 workers
        results = batch_classify_objects(
            vlm=vlm,
            entries=entries,
            llm=llm,
            image_base_dir=Path("data/images"),
            max_workers=4,
            output_key="vehicle_type",
            on_progress=lambda id, resp: print(f"Processed {id}")
        )
        ```
    """
    # Normalize processed_ids
    processed_ids = processed_ids or set()

    # Filter out already processed entries
    entries_to_process = [
        e for e in entries if e.get("id", "unknown") not in processed_ids
    ]
    skipped_count = len(entries) - len(entries_to_process)

    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already processed entries")

    logger.info(f"Starting batch classification for {len(entries_to_process)} entries")
    logger.info(
        f"Configuration: invoke_kwargs={invoke_kwargs}, max_workers={max_workers or 'sequential'}, output_key={output_key}"
    )
    if image_base_dir:
        logger.info(f"Image base directory: {image_base_dir}")

    # Decide between sequential and parallel processing
    use_parallel = max_workers is not None and max_workers > 1

    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers")
        return _process_parallel(
            vlm=vlm,
            entries=entries_to_process,
            llm=llm,
            image_base_dir=image_base_dir,
            system_prompt=system_prompt,
            invoke_kwargs=invoke_kwargs,
            on_progress=on_progress,
            on_error=on_error,
            on_result=on_result,
            on_prediction=on_prediction,
            max_workers=max_workers,
            max_retries=max_retries,
            output_key=output_key,
            token_tracker=token_tracker,
            unknown_sentinel=unknown_sentinel,
        )
    else:
        logger.info("Using sequential processing")
        return _process_sequential(
            vlm=vlm,
            entries=entries_to_process,
            llm=llm,
            image_base_dir=image_base_dir,
            system_prompt=system_prompt,
            invoke_kwargs=invoke_kwargs,
            on_progress=on_progress,
            on_error=on_error,
            on_result=on_result,
            on_prediction=on_prediction,
            max_retries=max_retries,
            output_key=output_key,
            token_tracker=token_tracker,
            unknown_sentinel=unknown_sentinel,
        )


def _extract_images_from_entry(
    entry: dict[str, Any],
) -> list[str | Path | PILImage.Image]:
    """Extract images from dataset entry supporting multiple formats.

    Supports:
    1. Old format: entry["images"] as list of paths
    2. Old format: entry["image_path"] as single path
    3. New format: entry["media"]["images"] as list of dicts with "path" key

    Args:
        entry: Dataset entry

    Returns:
        List of image paths or PIL Images
    """
    # Try old format first
    if "images" in entry and entry["images"]:
        return entry["images"]

    if "image_path" in entry and entry["image_path"]:
        return [entry["image_path"]]

    # Try new format: media.images[]
    if "media" in entry and isinstance(entry["media"], dict):
        media = entry["media"]
        if "images" in media and media["images"]:
            # Extract paths from image objects
            image_paths = []
            for img_obj in media["images"]:
                if isinstance(img_obj, dict) and "path" in img_obj:
                    image_paths.append(img_obj["path"])
                elif isinstance(img_obj, str | Path | PILImage.Image):
                    # Support mixed format
                    image_paths.append(img_obj)
            return image_paths

    return []


def _extract_image_metadata_from_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract image metadata from dataset entry supporting multiple formats.

    Returns metadata for each image, which may include vlm_prompt, view, camera, etc.

    Args:
        entry: Dataset entry

    Returns:
        List of metadata dicts (one per image)
    """
    # Try new format first: media.images[] with metadata
    if "media" in entry and isinstance(entry["media"], dict):
        media = entry["media"]
        if "images" in media and media["images"]:
            metadata_list = []
            for img_obj in media["images"]:
                if isinstance(img_obj, dict):
                    # Extract metadata from image object
                    metadata = img_obj.get("metadata", {})
                    metadata_list.append(metadata)
                else:
                    # No metadata for this image
                    metadata_list.append({})
            return metadata_list

    # Try old format: entry["image_metadata"]
    if "image_metadata" in entry and entry["image_metadata"]:
        return entry["image_metadata"]

    return []


def _extract_text_from_entry(entry: dict[str, Any]) -> str:
    """Extract prompt text from dataset entry supporting multiple formats.

    Supports:
    - Old format: entry["text"]
    - New format: entry["user_prompt"]

    Args:
        entry: Dataset entry

    Returns:
        Prompt text string
    """
    # Try old format first for backward compatibility
    if "text" in entry:
        return entry["text"]

    # Try new format
    if "user_prompt" in entry:
        return entry["user_prompt"]

    return ""


def _process_sequential(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None,
    system_prompt: str | None,
    invoke_kwargs: dict[str, Any] | None,
    on_progress: Any | None,
    on_error: Any | None,
    on_result: Any | None,
    on_prediction: Any | None,
    max_retries: int,
    output_key: str,
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> list[dict[str, Any]]:
    """Process entries sequentially (original behavior)."""
    results = []

    # Running timing stats for ETA
    total_assignment_seconds = 0.0
    timed_assignments = 0

    def _format_duration(total_seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        seconds_int = int(total_seconds)
        hours = seconds_int // 3600
        minutes = (seconds_int % 3600) // 60
        seconds = seconds_int % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    for idx, entry in enumerate(entries, 1):
        entry_id = entry.get("id", "unknown")
        logger.debug(f"Processing entry {idx}/{len(entries)}: {entry_id}")

        try:
            # Prepare images - can be paths or PIL Images
            # Support multiple dataset formats
            images = _extract_images_from_entry(entry)
            logger.debug(f"Entry {entry_id} has {len(images)} images")

            processed_images = []
            missing_images = []
            invalid_images = []

            for img in images:
                if isinstance(img, PILImage.Image):
                    # Already a PIL Image, use as-is
                    processed_images.append(img)
                elif isinstance(img, str | Path):
                    # It's a path, process it
                    if image_base_dir and not Path(img).is_absolute():
                        img_path = image_base_dir / img
                    else:
                        img_path = Path(img)

                    # Check if file exists
                    if not img_path.exists():
                        missing_images.append(str(img_path))
                    else:
                        processed_images.append(img_path)
                else:
                    invalid_images.append(f"{img} (type: {type(img).__name__})")

            # Consolidate all image errors into a single error message
            error_messages = []
            if invalid_images:
                error_messages.append(
                    f"Unsupported image types: {invalid_images}. Expected str, Path, or PIL Image"
                )
            if missing_images:
                error_messages.append(f"Missing images: {missing_images}")

            if error_messages:
                error_msg = "; ".join(error_messages)
                logger.error(f"Entry {entry_id}: {error_msg}")
                if on_error:
                    on_error(entry_id, error_msg)
                error_result = {
                    "id": entry_id,
                    "vlm_response": None,
                    "status": "error",
                    "error": error_msg,
                }
                results.append(error_result)
                # Emit result callback for error cases
                if on_result:
                    try:
                        on_result(error_result, entry)
                    except Exception as cb_e:
                        logger.warning(
                            f"on_result callback failed for {entry_id}: {cb_e}"
                        )
                continue

            # Get text prompt (supports both old and new format)
            text = _extract_text_from_entry(entry)
            text_preview = text[:100] if text else ""
            logger.debug(f"Entry {entry_id} text: {text_preview}...")

            # Check if entry has its own system_prompt (per-entry override)
            entry_system_prompt = entry.get("system_prompt", system_prompt)

            # Get classification with progress indication
            percent = int((idx / max(len(entries), 1)) * 100)
            logger.info(
                f"Running VLM inference for entry {idx}/{len(entries)} ({percent}%) - {entry_id}"
            )
            start_time = perf_counter()
            # Get image prompts if available from metadata (supports both old and new format)
            image_prompts = None
            image_metadata = _extract_image_metadata_from_entry(entry)
            if image_metadata and len(image_metadata) == len(processed_images):
                image_prompts = []
                for meta in image_metadata:
                    if "vlm_prompt" in meta:
                        image_prompts.append(meta["vlm_prompt"])
                    else:
                        image_prompts = (
                            None  # If any image lacks a prompt, don't use prompts
                        )
                        break

            response = classify_object(
                vlm=vlm,
                text=text,
                images=processed_images,
                llm=llm,
                system_prompt=entry_system_prompt,
                invoke_kwargs=invoke_kwargs,
                image_prompts=image_prompts,
                max_retries=max_retries,
                output_key=output_key,
                token_tracker=token_tracker,
                unknown_sentinel=unknown_sentinel,
            )

            # Log response preview
            if isinstance(response, dict):
                response_preview = f"{output_key}='{response.get(output_key)}'"
            else:
                response_preview = response[:100] if response else "No response"
            logger.info(f"Entry {entry_id} response: {response_preview}")

            # Timing and ETA after each assignment
            elapsed_seconds = perf_counter() - start_time
            total_assignment_seconds += elapsed_seconds
            timed_assignments += 1
            avg_seconds = total_assignment_seconds / max(timed_assignments, 1)
            remaining = max(len(entries) - idx, 0)
            eta_seconds = avg_seconds * remaining
            finish_time_local = datetime.now() + timedelta(seconds=int(eta_seconds))
            logger.info(
                "Timing: last=%ss, avg=%ss, remaining about %s, ETA finish at %s",
                f"{elapsed_seconds:.1f}",
                f"{avg_seconds:.1f}",
                _format_duration(eta_seconds),
                finish_time_local.strftime("%Y-%m-%d %H:%M:%S"),
            )

            # Store result
            result = {
                "id": entry_id,
                "vlm_response": response,
                "status": "success",
            }
            results.append(result)

            # Call progress callback if provided
            if on_progress:
                on_progress(entry_id, response)

            # Emit prediction callback if provided
            if on_prediction:
                try:
                    on_prediction(entry_id, response)
                except Exception as cb_e:
                    logger.warning(
                        f"on_prediction callback failed for {entry_id}: {cb_e}"
                    )

            # Emit result callback if provided
            if on_result:
                try:
                    on_result(result, entry)
                except Exception as cb_e:
                    logger.warning(f"on_result callback failed for {entry_id}: {cb_e}")

        except Exception as e:
            raise_for_model_authentication(e)
            error_msg = str(e)
            logger.error(f"Entry {entry_id} failed: {error_msg}", exc_info=True)
            # Even on failure, record timing and print ETA if we started timing
            try:
                if "start_time" in locals():
                    elapsed_seconds = perf_counter() - start_time
                    total_assignment_seconds += elapsed_seconds
                    timed_assignments += 1
                    avg_seconds = total_assignment_seconds / max(timed_assignments, 1)
                    remaining = max(len(entries) - idx, 0)
                    eta_seconds = avg_seconds * remaining
                    finish_time_local = datetime.now() + timedelta(
                        seconds=int(eta_seconds)
                    )
                    logger.info(
                        "Timing: last=%ss, avg=%ss, remaining≈%s, ETA finish at %s",
                        f"{elapsed_seconds:.1f}",
                        f"{avg_seconds:.1f}",
                        _format_duration(eta_seconds),
                        finish_time_local.strftime("%Y-%m-%d %H:%M:%S"),
                    )
            except Exception:  # best-effort timing; don't block on ETA errors
                pass
            if on_error:
                on_error(entry_id, error_msg)

            error_result = {
                "id": entry_id,
                "vlm_response": None,
                "status": "error",
                "error": error_msg,
            }
            results.append(error_result)
            # Emit result callback for error cases
            if on_result:
                try:
                    on_result(error_result, entry)
                except Exception as cb_e:
                    logger.warning(f"on_result callback failed for {entry_id}: {cb_e}")

    # Log summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(
        f"Sequential processing complete: {successful} successful, {failed} failed out of {len(entries)} total"
    )

    return results


def _process_parallel(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None,
    system_prompt: str | None,
    invoke_kwargs: dict[str, Any] | None,
    on_progress: Any | None,
    on_error: Any | None,
    on_result: Any | None,
    on_prediction: Any | None,
    max_workers: int,
    max_retries: int,
    output_key: str,
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> list[dict[str, Any]]:
    """Process entries in parallel using ThreadPoolExecutor."""
    results = []
    auth_abort = Event()

    # Thread-safe statistics
    stats_lock = Lock()
    total_assignment_seconds = 0.0
    timed_assignments = 0
    completed_count = 0

    def _format_duration(total_seconds: float) -> str:
        """Format seconds as H:MM:SS."""
        seconds_int = int(total_seconds)
        hours = seconds_int // 3600
        minutes = (seconds_int % 3600) // 60
        seconds = seconds_int % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _process_entry(entry: dict[str, Any]) -> dict[str, Any]:
        """Process a single entry (executed in thread pool)."""
        nonlocal total_assignment_seconds, timed_assignments, completed_count

        if auth_abort.is_set():
            raise ModelAuthenticationFailure()

        entry_id = entry.get("id", "unknown")
        logger.debug(f"Processing entry: {entry_id}")

        try:
            # Prepare images - can be paths or PIL Images
            # Support multiple dataset formats
            images = _extract_images_from_entry(entry)
            logger.debug(f"Entry {entry_id} has {len(images)} images")

            processed_images = []
            missing_images = []
            invalid_images = []

            for img in images:
                if isinstance(img, PILImage.Image):
                    # Already a PIL Image, use as-is
                    processed_images.append(img)
                elif isinstance(img, str | Path):
                    # It's a path, process it
                    if image_base_dir and not Path(img).is_absolute():
                        img_path = image_base_dir / img
                    else:
                        img_path = Path(img)

                    # Check if file exists
                    if not img_path.exists():
                        missing_images.append(str(img_path))
                    else:
                        processed_images.append(img_path)
                else:
                    invalid_images.append(f"{img} (type: {type(img).__name__})")

            # Consolidate all image errors into a single error message
            error_messages = []
            if invalid_images:
                error_messages.append(
                    f"Unsupported image types: {invalid_images}. Expected str, Path, or PIL Image"
                )
            if missing_images:
                error_messages.append(f"Missing images: {missing_images}")

            if error_messages:
                error_msg = "; ".join(error_messages)
                logger.error(f"Entry {entry_id}: {error_msg}")
                if on_error:
                    on_error(entry_id, error_msg)
                error_result = {
                    "id": entry_id,
                    "vlm_response": None,
                    "status": "error",
                    "error": error_msg,
                }
                # Emit result callback for error cases
                if on_result:
                    try:
                        on_result(error_result, entry)
                    except Exception as cb_e:
                        logger.warning(
                            f"on_result callback failed for {entry_id}: {cb_e}"
                        )
                return error_result

            # Get text prompt (supports both old and new format)
            text = _extract_text_from_entry(entry)
            text_preview = text[:100] if text else ""
            logger.debug(f"Entry {entry_id} text: {text_preview}...")

            # Check if entry has its own system_prompt (per-entry override)
            entry_system_prompt = entry.get("system_prompt", system_prompt)

            # Get classification
            logger.info(f"Running VLM inference for {entry_id}")
            start_time = perf_counter()
            # Get image prompts if available from metadata (supports both old and new format)
            image_prompts = None
            image_metadata = _extract_image_metadata_from_entry(entry)
            if image_metadata and len(image_metadata) == len(processed_images):
                image_prompts = []
                for meta in image_metadata:
                    if "vlm_prompt" in meta:
                        image_prompts.append(meta["vlm_prompt"])
                    else:
                        image_prompts = (
                            None  # If any image lacks a prompt, don't use prompts
                        )
                        break

            if auth_abort.is_set():
                raise ModelAuthenticationFailure()
            response = classify_object(
                vlm=vlm,
                text=text,
                images=processed_images,
                llm=llm,
                system_prompt=entry_system_prompt,
                invoke_kwargs=invoke_kwargs,
                image_prompts=image_prompts,
                max_retries=max_retries,
                output_key=output_key,
                token_tracker=token_tracker,
                unknown_sentinel=unknown_sentinel,
            )

            # Log response preview
            if isinstance(response, dict):
                response_preview = f"{output_key}='{response.get(output_key)}'"
            else:
                response_preview = response[:100] if response else "No response"
            logger.info(f"Entry {entry_id} response: {response_preview}")

            # Update timing statistics (thread-safe)
            elapsed_seconds = perf_counter() - start_time
            with stats_lock:
                total_assignment_seconds += elapsed_seconds
                timed_assignments += 1
                completed_count += 1
                avg_seconds = total_assignment_seconds / max(timed_assignments, 1)
                remaining = max(len(entries) - completed_count, 0)
                eta_seconds = avg_seconds * remaining
                finish_time_local = datetime.now() + timedelta(seconds=int(eta_seconds))
                logger.info(
                    "Timing: last=%ss, avg=%ss, completed=%d/%d, remaining≈%s, ETA finish at %s",
                    f"{elapsed_seconds:.1f}",
                    f"{avg_seconds:.1f}",
                    completed_count,
                    len(entries),
                    _format_duration(eta_seconds),
                    finish_time_local.strftime("%Y-%m-%d %H:%M:%S"),
                )

            # Store result
            result = {
                "id": entry_id,
                "vlm_response": response,
                "status": "success",
            }

            # Call progress callback if provided
            if on_progress:
                on_progress(entry_id, response)

            # Emit prediction callback if provided
            if on_prediction:
                try:
                    on_prediction(entry_id, response)
                except Exception as cb_e:
                    logger.warning(
                        f"on_prediction callback failed for {entry_id}: {cb_e}"
                    )

            # Emit result callback if provided
            if on_result:
                try:
                    on_result(result, entry)
                except Exception as cb_e:
                    logger.warning(f"on_result callback failed for {entry_id}: {cb_e}")

            return result

        except Exception as e:
            try:
                raise_for_model_authentication(e)
            except ModelAuthenticationFailure:
                auth_abort.set()
                raise
            error_msg = str(e)
            logger.error(f"Entry {entry_id} failed: {error_msg}", exc_info=True)

            # Update timing even on failure
            if "start_time" in locals():
                try:
                    elapsed_seconds = perf_counter() - start_time
                    with stats_lock:
                        total_assignment_seconds += elapsed_seconds
                        timed_assignments += 1
                        completed_count += 1
                except Exception:
                    pass

            if on_error:
                on_error(entry_id, error_msg)

            error_result = {
                "id": entry_id,
                "vlm_response": None,
                "status": "error",
                "error": error_msg,
            }

            # Emit result callback for error cases
            if on_result:
                try:
                    on_result(error_result, entry)
                except Exception as cb_e:
                    logger.warning(f"on_result callback failed for {entry_id}: {cb_e}")

            return error_result

    # Process entries in parallel using ThreadPoolExecutor
    auth_failure: ModelAuthenticationFailure | None = None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(_process_entry, entry): entry for entry in entries
        }

        # Collect results as they complete
        for future in as_completed(future_to_entry):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                try:
                    raise_for_model_authentication(e)
                except ModelAuthenticationFailure as error:
                    auth_failure = error
                    for sibling in future_to_entry:
                        if sibling is not future:
                            sibling.cancel()
                    break
                entry = future_to_entry[future]
                entry_id = entry.get("id", "unknown")
                logger.error(
                    f"Unexpected error processing {entry_id}: {e}", exc_info=True
                )
                error_result = {
                    "id": entry_id,
                    "vlm_response": None,
                    "status": "error",
                    "error": str(e),
                }
                results.append(error_result)

    if auth_failure is not None:
        raise auth_failure from None

    # Log summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(
        f"Parallel processing complete: {successful} successful, {failed} failed out of {len(entries)} total (using {max_workers} workers)"
    )

    return results


# ============================================================================
# Multi-object (multi-prim) classification in a single VLM call
# ============================================================================


def classify_objects_multi_prim(
    vlm: BaseVisionLanguageModel,
    object_ids: list[str],
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    output_key: str = "class",
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Classify multiple objects in a single VLM call.

    Sends one VLM request with images for N objects and parses a JSON response
    mapping each object_id to its classification.

    Expected VLM response format:
        {
            "object_id_1": {"<output_key>": "value"},
            "object_id_2": {"<output_key>": "value"},
            ...
        }

    Args:
        vlm: Vision-Language Model instance
        object_ids: List of object identifiers expected in the response
        text: Combined user prompt with per-object context and image layout
        images: Combined list of images (reference + per-object images)
        llm: LLM for fallback parsing
        system_prompt: System prompt instructing multi-object classification
        invoke_kwargs: Additional kwargs for VLM (temperature, max_tokens, etc.)
        image_prompts: Optional per-image captions
        max_retries: Max retry attempts for VLM/LLM calls
        output_key: Key name for the classification result (default: "class")
        token_tracker: Optional token usage tracker
        unknown_sentinel: Optional explicit sentinel value to preserve during
            fallback parsing when an object is unknown/unclassifiable

    Returns:
        Dict mapping object_id to classification result dicts. Structured
        metadata from nested VLM responses is preserved.
        Only successfully parsed objects are included. Missing objects indicate
        partial failure — the caller should handle re-queuing.
    """
    # Extract temperature and max_tokens from invoke_kwargs if present
    temperature = None
    max_tokens = None
    if invoke_kwargs:
        temperature = invoke_kwargs.get("temperature")
        max_tokens = invoke_kwargs.get(
            "max_tokens",
            invoke_kwargs.get("max_completion_tokens"),
        )

    if system_prompt is None:
        system_prompt = (
            "You are an expert at identifying objects and their properties. "
            "Analyze the images carefully and classify each object."
        )

    if not images or len(images) == 0:
        raise ValueError("classify_objects_multi_prim called with empty images list")

    logger.info(
        f"Running multi-object classification for {len(object_ids)} objects "
        f"with {len(images)} images"
    )

    # --- VLM call with retry ---
    vlm_response = ""
    current_max_tokens = max_tokens

    for attempt in range(max_retries):
        try:
            if image_prompts and len(image_prompts) == len(images):
                image_caption_pairs = list(zip(image_prompts, images, strict=False))
                vlm_response = vlm.generate_with_image_caption_pairs(
                    image_caption_pairs=image_caption_pairs,
                    final_prompt=text,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                )
            else:
                if image_prompts and len(image_prompts) != len(images):
                    logger.warning(
                        f"Image prompts count ({len(image_prompts)}) doesn't match "
                        f"images count ({len(images)}), ignoring prompts"
                    )
                vlm_response = vlm.generate(
                    prompt=text,
                    images=images,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=current_max_tokens,
                )

            if token_tracker is not None and vlm.last_token_usage is not None:
                token_tracker.add_usage(vlm.last_token_usage)

            if vlm_response and vlm_response.strip():
                logger.debug(
                    f"Multi-prim VLM response received on attempt "
                    f"{attempt + 1}/{max_retries}"
                )
                break
            else:
                if attempt < max_retries - 1:
                    if current_max_tokens:
                        previous_tokens = current_max_tokens
                        current_max_tokens = current_max_tokens * 2
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}. "
                            f"Doubling max_tokens from {previous_tokens} to {current_max_tokens}."
                        )
                    else:
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}, retrying..."
                        )
                    import time

                    time.sleep(get_fibonacci_delay(attempt, base_delay=1.0))
                else:
                    logger.error(f"Empty VLM response after {max_retries} attempts")
                    vlm_response = ""

        except Exception as e:
            raise_for_model_authentication(e)
            logger.error(
                f"VLM inference error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt < max_retries - 1:
                import time

                time.sleep(get_fibonacci_delay(attempt, base_delay=1.0))
            else:
                raise

    # --- Parse multi-object response ---
    return _parse_multi_prim_response(
        vlm_response=vlm_response,
        object_ids=object_ids,
        output_key=output_key,
        llm=llm,
        system_prompt=system_prompt,
        text=text,
        max_retries=max_retries,
        unknown_sentinel=unknown_sentinel,
    )


def _parse_multi_prim_response(
    vlm_response: str,
    object_ids: list[str],
    output_key: str,
    llm: BaseChatModel,
    system_prompt: str,
    text: str,
    max_retries: int = 3,
    unknown_sentinel: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse a multi-object VLM response into per-object results.

    Tries multiple strategies:
    1. Extract <answer> block, parse as JSON dict
    2. Extract JSON from full response
    3. LLM fallback parsing

    Args:
        vlm_response: Raw VLM response text
        object_ids: Expected object IDs
        output_key: Classification key (e.g. "material")
        llm: LLM for fallback parsing
        system_prompt: Original system prompt (for LLM context)
        text: Original user prompt (for LLM context)
        max_retries: Max retries for LLM fallback
        unknown_sentinel: Optional explicit sentinel value to preserve during
            fallback parsing when an object is unknown/unclassifiable

    Returns:
        Dict mapping object_id to classification result dicts. Structured
        metadata from nested VLM responses is preserved.
    """
    results: dict[str, dict[str, Any]] = {}
    object_id_set = set(object_ids)

    def _coerce_result_payload(value: Any) -> dict[str, Any] | None:
        """Coerce one parsed object value into a classification payload."""
        if isinstance(value, str):
            return {
                output_key: value,
                "original_response": vlm_response,
            }
        if not isinstance(value, dict):
            return None

        material_value = None
        payload: dict[str, Any] = dict(value)

        nested_materials = value.get("materials")
        if isinstance(nested_materials, dict):
            if output_key in nested_materials:
                material_value = nested_materials.get(output_key)
            elif "material" in nested_materials:
                material_value = nested_materials.get("material")
            reason = nested_materials.get("reason")
            if isinstance(reason, str) and reason.strip() and "reason" not in payload:
                payload["reason"] = reason
        elif isinstance(nested_materials, str):
            material_value = nested_materials

        if material_value is None:
            if output_key in value:
                material_value = value.get(output_key)
            elif "material" in value:
                material_value = value.get("material")
            else:
                material_value = extract_material_from_json(value)

        reason = value.get("reason")
        if isinstance(reason, str) and reason.strip():
            payload["reason"] = reason

        if not isinstance(material_value, str) or not material_value.strip():
            return None

        _rename_legacy_material_key(
            payload,
            output_key=output_key,
            value=material_value,
        )
        _normalize_unknown_sentinel_result(
            payload,
            output_key=output_key,
            unknown_sentinel=unknown_sentinel,
        )
        payload["original_response"] = vlm_response
        return payload

    def _object_id_from_record(record: dict[str, Any]) -> str | None:
        """Extract an expected object ID from a record-shaped prediction."""
        for id_key in ("id", "object_id", "prim_path", "path"):
            object_id = record.get(id_key)
            if isinstance(object_id, str) and object_id in object_id_set:
                return object_id
        return None

    def _try_extract_from_parsed(data: Any) -> dict[str, dict[str, Any]]:
        """Try to extract per-object results from parsed JSON-like data."""
        extracted: dict[str, dict[str, Any]] = {}

        if isinstance(data, list):
            for item in data:
                extracted.update(_try_extract_from_parsed(item))
            return extracted

        if not isinstance(data, dict):
            return extracted

        record_object_id = _object_id_from_record(data)
        if record_object_id:
            payload = _coerce_result_payload(data)
            if payload:
                extracted[record_object_id] = payload

        for key, value in data.items():
            if key in object_id_set:
                payload = _coerce_result_payload(value)
                if payload:
                    extracted[key] = payload

        for container_key in ("predictions", "results", "items", "objects"):
            container = data.get(container_key)
            if isinstance(container, dict | list):
                extracted.update(_try_extract_from_parsed(container))

        return extracted

    # Strategy 1: <answer> block
    answer_content = extract_answer_block(vlm_response)
    if answer_content:
        logger.debug("Found <answer> block in multi-prim response")
        parsed = extract_json_from_llm_response(answer_content)
        if parsed:
            results = _try_extract_from_parsed(parsed)
            if results:
                logger.info(
                    f"Parsed {len(results)}/{len(object_ids)} objects from <answer> block"
                )
                return results
        # Try direct JSON parse of answer content
        try:
            parsed = json.loads(answer_content)
            results = _try_extract_from_parsed(parsed)
            if results:
                logger.info(
                    f"Parsed {len(results)}/{len(object_ids)} objects from <answer> JSON"
                )
                return results
        except json.JSONDecodeError:
            pass

    # Strategy 2: Extract JSON from full response
    logger.debug("Trying to extract JSON from full multi-prim response")
    parsed = extract_json_from_llm_response(vlm_response)
    if parsed:
        results = _try_extract_from_parsed(parsed)
        if results:
            logger.info(
                f"Parsed {len(results)}/{len(object_ids)} objects from full response"
            )
            return results

    # Strategy 2b: Try to find the largest JSON object in the response
    # (handles cases where the VLM wraps the answer in extra text)
    json_objects = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", vlm_response)
    for json_str in sorted(json_objects, key=len, reverse=True):
        try:
            parsed = json.loads(json_str)
            results = _try_extract_from_parsed(parsed)
            if results:
                logger.info(
                    f"Parsed {len(results)}/{len(object_ids)} objects from "
                    f"embedded JSON object"
                )
                return results
        except json.JSONDecodeError:
            continue

    # Strategy 3: LLM fallback
    logger.warning("Direct parsing failed for multi-prim response, using LLM fallback")

    ids_str = "\n".join(f"  - {oid}" for oid in object_ids)
    context_char_limit = 4000
    system_prompt_context = (
        system_prompt[:context_char_limit] + "..."
        if len(system_prompt) > context_char_limit
        else system_prompt
    )
    text_context = (
        text[:context_char_limit] + "..." if len(text) > context_char_limit else text
    )
    sentinel_instruction = ""
    if unknown_sentinel:
        sentinel_instruction = (
            f'\nConfigured sentinel value: "{unknown_sentinel}". '
            "When an object is explicitly unknown or unclassifiable, return "
            f'"{unknown_sentinel}" exactly for "{output_key}".'
        )
    parsing_prompt = f"""The VLM was asked to classify multiple objects and return a JSON mapping.
However, the response was not in the expected format.

Expected object IDs:
{ids_str}

Original system prompt (truncated if needed):
{system_prompt_context}

Original user prompt (truncated if needed):
{text_context}

VLM response:
{vlm_response}

Please extract or infer the classification for each object ID.
Return ONLY a JSON object with this exact structure:
{{
  "<object_id>": {{"{output_key}": "value"}},
  ...
}}

Use the available options from the original system prompt. Preserve any explicit
sentinel or unknown value from the VLM response exactly. Every object ID listed
above MUST appear in your response. If you cannot determine a value and the
original prompt defines an unknown/unclassified sentinel, use that sentinel;
otherwise use your best guess from the available options.
{sentinel_instruction}
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        parser_system_prompt = (
            "You are an intelligent parser. Extract structured JSON from "
            "unstructured VLM output. Always return valid JSON."
        )
        messages = [
            SystemMessage(content=parser_system_prompt),
            HumanMessage(content=parsing_prompt),
        ]

        for llm_attempt in range(max_retries):
            try:
                parsed_text = _invoke_parser_model_sync(
                    llm,
                    messages=messages,
                    parsing_prompt=parsing_prompt,
                    parser_system_prompt=parser_system_prompt,
                    max_tokens=1024,
                )
                if not parsed_text or not parsed_text.strip():
                    if llm_attempt < max_retries - 1:
                        import time

                        time.sleep(get_fibonacci_delay(llm_attempt, base_delay=0.5))
                        continue
                    break

                # Try to parse LLM response
                answer_block = extract_answer_block(parsed_text)
                text_to_parse = answer_block if answer_block else parsed_text

                parsed = extract_json_from_llm_response(text_to_parse)
                if not parsed:
                    try:
                        parsed = json.loads(text_to_parse)
                    except json.JSONDecodeError:
                        pass

                if parsed:
                    results = _try_extract_from_parsed(parsed)
                    if results:
                        logger.info(
                            f"LLM fallback parsed {len(results)}/{len(object_ids)} objects"
                        )
                        return results

                if llm_attempt < max_retries - 1:
                    import time

                    time.sleep(get_fibonacci_delay(llm_attempt, base_delay=0.5))

            except Exception as e:
                raise_for_model_authentication(e)
                logger.error(f"LLM parsing attempt {llm_attempt + 1} failed: {e}")
                if llm_attempt < max_retries - 1:
                    import time

                    time.sleep(get_fibonacci_delay(llm_attempt, base_delay=0.5))

    except Exception as e:
        raise_for_model_authentication(e)
        logger.error(f"LLM fallback failed entirely: {e}")

    logger.error(
        f"All parsing strategies failed for multi-prim response. "
        f"Returning empty results for {len(object_ids)} objects."
    )
    return results


# ============================================================================
# Async variants
# ============================================================================


async def async_classify_object(
    vlm: BaseVisionLanguageModel,
    text: str,
    images: list[str | Path | PILImage.Image],
    llm: BaseChatModel,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    image_prompts: list[str] | None = None,
    max_retries: int = 3,
    output_key: str = "class",
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> dict[str, Any]:
    """Classify an object using Vision-Language Model asynchronously.

    Async version of classify_object() that uses vlm.agenerate() for true
    async I/O instead of blocking the event loop.

    Args:
        vlm: Vision-Language Model instance to use for inference
        text: Context text containing object description and available classes
        images: List of images as file paths (str/Path) or PIL Image objects
        llm: LLM for parsing VLM response into structured format (fallback only)
        system_prompt: Optional custom system prompt (uses default if None)
        invoke_kwargs: Additional kwargs to pass to VLM (e.g., temperature, max_tokens)
        image_prompts: Optional list of prompts/captions for each image
        max_retries: Maximum number of retry attempts for VLM/LLM calls (default: 3)
        output_key: Key name for the classification result in output dict (default: "class")
        token_tracker: Optional TokenTracker to collect usage statistics
        unknown_sentinel: Optional explicit sentinel value to preserve when the
            VLM says the object is unknown/unclassifiable

    Returns:
        Dict with output_key and "original_response" keys
    """
    # Extract temperature and max_tokens from invoke_kwargs if present
    temperature = None
    max_tokens = None
    if invoke_kwargs:
        temperature = invoke_kwargs.get("temperature")
        mt = invoke_kwargs.get("max_tokens")
        max_tokens = (
            mt if mt is not None else invoke_kwargs.get("max_completion_tokens")
        )
    # Default system prompt if not provided
    if system_prompt is None:
        system_prompt = (
            "You are an expert at identifying objects and their properties. "
            "Analyze the images carefully and provide clear reasoning for your "
            "classification."
        )

    prompt = text

    # Check for empty or None images
    if not images or len(images) == 0:
        error_msg = (
            f"async_classify_object called with empty or None images list! "
            f"images={images}. The VLM will not be able to analyze anything."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Running async classification with {len(images)} images")

    # Run VLM inference with retry logic for empty responses
    vlm_response = ""
    current_max_tokens = max_tokens
    timeout_seconds = _get_vlm_generate_timeout_seconds()

    for attempt in range(max_retries):
        try:
            if image_prompts and len(image_prompts) == len(images):
                logger.debug("Using image-caption pairs for async VLM inference")
                image_caption_pairs = list(zip(image_prompts, images, strict=False))
                vlm_response = await _call_async_with_timeout(
                    vlm.agenerate_with_image_caption_pairs(
                        image_caption_pairs=image_caption_pairs,
                        final_prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                    ),
                    timeout_seconds=timeout_seconds,
                    operation_name="VLM agenerate_with_image_caption_pairs",
                )
            else:
                if image_prompts and len(image_prompts) != len(images):
                    logger.warning(
                        f"Image prompts count ({len(image_prompts)}) doesn't match "
                        f"images count ({len(images)}), ignoring prompts"
                    )
                vlm_response = await _call_async_with_timeout(
                    vlm.agenerate(
                        prompt=prompt,
                        images=images,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=current_max_tokens,
                    ),
                    timeout_seconds=timeout_seconds,
                    operation_name="VLM agenerate",
                )

            # Track token usage if tracker provided
            if token_tracker is not None and vlm.last_token_usage is not None:
                token_tracker.add_usage(vlm.last_token_usage)

            # Check if response is empty or just whitespace
            if vlm_response and vlm_response.strip():
                logger.debug(
                    f"VLM response received on attempt {attempt + 1}/{max_retries}"
                )
                break
            else:
                if attempt < max_retries - 1:
                    if current_max_tokens:
                        previous_tokens = current_max_tokens
                        current_max_tokens = current_max_tokens * 2
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}. "
                            f"Doubling max_tokens from {previous_tokens} to {current_max_tokens}."
                        )
                    else:
                        logger.warning(
                            f"Empty VLM response on attempt {attempt + 1}/{max_retries}, retrying..."
                        )

                    retry_delay = get_fibonacci_delay(attempt, base_delay=1.0)
                    logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"Empty VLM response after {max_retries} attempts")
                    vlm_response = ""

        except Exception as e:
            raise_for_model_authentication(e)
            logger.error(
                f"VLM inference error on attempt {attempt + 1}/{max_retries}: {e}"
            )
            if attempt < max_retries - 1:
                retry_delay = get_fibonacci_delay(attempt, base_delay=1.0)
                logger.info(f"Retrying in {retry_delay:.1f} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                raise

    result = _parse_single_result_from_response_text(
        vlm_response,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    if result:
        _normalize_unknown_sentinel_result(
            result,
            output_key=output_key,
            unknown_sentinel=unknown_sentinel,
        )
        result["original_response"] = vlm_response
        return result

    sentinel_result = _explicit_unknown_sentinel_result(
        vlm_response,
        output_key=output_key,
        unknown_sentinel=unknown_sentinel,
    )
    if sentinel_result:
        logger.info(
            f"Preserving explicit unknown sentinel for {output_key}: "
            f"'{unknown_sentinel}'"
        )
        return sentinel_result

    # LLM fallback parsing - use asyncio.to_thread since LLM may not have ainvoke
    logger.debug("Direct JSON extraction failed, using LLM to parse VLM response")

    if unknown_sentinel:
        unknown_instruction = (
            f'\n- If the VLM response is exactly "{unknown_sentinel}", '
            f'preserve that value exactly as "{output_key}".'
        )
        prediction_instruction = (
            "- Preserve explicit configured sentinel values from the VLM response. "
            "If there is no visible evidence and the original prompt defines the "
            f'sentinel "{unknown_sentinel}", return that sentinel instead of guessing.'
        )
        value_instruction = (
            "The value must match exactly as it appears in the options list, or "
            f'must be "{unknown_sentinel}" when the response is explicitly unknown.'
        )
        parser_system_choice = (
            "Choose a real option from the available list unless the VLM explicitly "
            f'returned the configured sentinel "{unknown_sentinel}".'
        )
    else:
        unknown_instruction = ""
        prediction_instruction = (
            '- DO NOT return "unknown" - make an informed choice from the '
            "available options"
        )
        value_instruction = (
            "The value must match exactly as it appears in the options list. "
            'Always choose a real option from the list, never "unknown".'
        )
        parser_system_choice = "Always choose a real option from the available list."

    parsing_prompt = f"""CONTEXT: You are acting as an intelligent fallback because the VLM failed to return properly structured JSON output.

The VLM was asked to analyze images and select a class/label, but its response was not in the expected JSON format. You have two options:

1. **EXTRACT**: If the VLM response contains a clear selection, extract it
2. **PREDICT**: If the VLM response is unclear/invalid, use the same context to make your own prediction

ORIGINAL VLM SYSTEM PROMPT (with classes list):
{system_prompt}

ORIGINAL VLM USER PROMPT/CONTEXT (describes the object):
{text}

VLM'S ACTUAL RESPONSE (unstructured):
{vlm_response}

YOUR TASK:
**Step 1 - Try to Extract:**
- Look for <answer> blocks first, then <reasoning> blocks, then overall response
- If VLM mentioned multiple options, choose the most specific/final one
- The value MUST be from the available options list in the system prompt
{unknown_instruction}

**Step 2 - If Extract Fails, Make Your Own Prediction:**
- Use the VLM system prompt (classification expertise) + user context (object description)
- Select the BEST MATCH from the available options list
{prediction_instruction}

OUTPUT REQUIREMENTS:
Return ONLY a JSON object with this exact structure:
{{
    "{output_key}": "selected_value"
}}

{value_instruction}
"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        parser_system_prompt = (
            "You are an intelligent LLM fallback for classification. A Vision-Language Model (VLM) "
            "failed to return structured output. You can either extract the value from the VLM response, "
            "OR if that's unclear, make your own informed prediction using the same context. "
            f"{parser_system_choice}"
        )
        messages = [
            SystemMessage(content=parser_system_prompt),
            HumanMessage(content=parsing_prompt),
        ]

        parsed_response = ""

        for llm_attempt in range(max_retries):
            try:
                parsed_response = await _invoke_parser_model_async(
                    llm,
                    messages=messages,
                    parsing_prompt=parsing_prompt,
                    parser_system_prompt=parser_system_prompt,
                    max_tokens=256,
                )

                if parsed_response and parsed_response.strip():
                    break
                else:
                    if llm_attempt < max_retries - 1:
                        llm_retry_delay = get_fibonacci_delay(
                            llm_attempt, base_delay=0.5
                        )
                        logger.warning(
                            f"Empty LLM response on attempt {llm_attempt + 1}/{max_retries}, "
                            f"retrying in {llm_retry_delay:.1f}s..."
                        )
                        await asyncio.sleep(llm_retry_delay)
                    else:
                        logger.error(
                            f"Empty LLM parsing response after {max_retries} attempts"
                        )
                        parsed_response = ""

            except Exception as e:
                raise_for_model_authentication(e)
                logger.error(
                    f"LLM parsing error on attempt {llm_attempt + 1}/{max_retries}: {e}"
                )
                if llm_attempt < max_retries - 1:
                    llm_retry_delay = get_fibonacci_delay(llm_attempt, base_delay=0.5)
                    await asyncio.sleep(llm_retry_delay)
                else:
                    raise

        result = _parse_single_result_from_response_text(
            parsed_response,
            output_key=output_key,
            unknown_sentinel=unknown_sentinel,
        )

        if result:
            _normalize_unknown_sentinel_result(
                result,
                output_key=output_key,
                unknown_sentinel=unknown_sentinel,
            )
            result["original_response"] = vlm_response
            return result
        else:
            return {
                output_key: "Unable to parse",
                "original_response": vlm_response,
            }

    except Exception as e:
        raise_for_model_authentication(e)
        logger.error(f"Error parsing VLM response with LLM: {e}")
        return {
            output_key: "Error during parsing",
            "original_response": vlm_response,
        }


async def async_batch_classify_objects(
    vlm: BaseVisionLanguageModel,
    entries: list[dict[str, Any]],
    llm: BaseChatModel,
    image_base_dir: Path | None = None,
    system_prompt: str | None = None,
    invoke_kwargs: dict[str, Any] | None = None,
    on_progress: Any | None = None,
    on_error: Any | None = None,
    processed_ids: set[str] | None = None,
    on_result: Any | None = None,
    on_prediction: Any | None = None,
    max_workers: int | None = None,
    max_retries: int = 3,
    output_key: str = "class",
    token_tracker: TokenTracker | None = None,
    unknown_sentinel: str | None = None,
) -> list[dict[str, Any]]:
    """Process multiple classification tasks using asyncio.gather with semaphore.

    Async version of batch_classify_objects() that uses asyncio.gather() with
    a semaphore for concurrency control instead of ThreadPoolExecutor.

    Args:
        Same as batch_classify_objects().

    Returns:
        List of dictionaries containing:
            - id: Entry identifier
            - vlm_response: Classification response
            - status: "success" or "error"
            - error: Error message (if status is "error")
    """
    processed_ids = processed_ids or set()

    entries_to_process = [
        e for e in entries if e.get("id", "unknown") not in processed_ids
    ]
    skipped_count = len(entries) - len(entries_to_process)

    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already processed entries")

    logger.info(
        f"Starting async batch classification for {len(entries_to_process)} entries"
    )
    logger.info(
        f"Configuration: invoke_kwargs={invoke_kwargs}, "
        f"max_workers={max_workers or 1}, output_key={output_key}"
    )

    # Semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers or 1)
    auth_abort = asyncio.Event()

    # Running timing stats (protected by lock for concurrent access)
    total_assignment_seconds = 0.0
    timed_assignments = 0
    stats_lock = asyncio.Lock()

    def _format_duration(total_seconds: float) -> str:
        seconds_int = int(total_seconds)
        hours = seconds_int // 3600
        minutes = (seconds_int % 3600) // 60
        seconds = seconds_int % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    async def _process_entry(entry: dict[str, Any], idx: int) -> dict[str, Any]:
        nonlocal total_assignment_seconds, timed_assignments

        async with semaphore:
            if auth_abort.is_set():
                raise ModelAuthenticationFailure()

            entry_id = entry.get("id", "unknown")
            logger.debug(
                f"Processing entry {idx}/{len(entries_to_process)}: {entry_id}"
            )

            try:
                # Prepare images
                images = _extract_images_from_entry(entry)
                processed_images: list[str | Path | PILImage.Image] = []
                missing_images: list[str] = []
                invalid_images: list[str] = []

                for img in images:
                    if isinstance(img, PILImage.Image):
                        processed_images.append(img)
                    elif isinstance(img, str | Path):
                        if image_base_dir and not Path(img).is_absolute():
                            img_path = image_base_dir / img
                        else:
                            img_path = Path(img)
                        if not img_path.exists():
                            missing_images.append(str(img_path))
                        else:
                            processed_images.append(img_path)
                    else:
                        invalid_images.append(f"{img} (type: {type(img).__name__})")

                error_messages = []
                if invalid_images:
                    error_messages.append(f"Unsupported image types: {invalid_images}")
                if missing_images:
                    error_messages.append(f"Missing images: {missing_images}")

                if error_messages:
                    error_msg = "; ".join(error_messages)
                    logger.error(f"Entry {entry_id}: {error_msg}")
                    if on_error:
                        on_error(entry_id, error_msg)
                    error_result: dict[str, Any] = {
                        "id": entry_id,
                        "vlm_response": None,
                        "status": "error",
                        "error": error_msg,
                    }
                    if on_result:
                        try:
                            on_result(error_result, entry)
                        except Exception as cb_e:
                            logger.warning(
                                f"on_result callback failed for {entry_id}: {cb_e}"
                            )
                    return error_result

                text = _extract_text_from_entry(entry)
                entry_system_prompt = entry.get("system_prompt", system_prompt)

                percent = int((idx / max(len(entries_to_process), 1)) * 100)
                logger.info(
                    f"Running async VLM inference for entry "
                    f"{idx}/{len(entries_to_process)} ({percent}%) - {entry_id}"
                )
                start_time = perf_counter()

                # Get image prompts if available
                image_prompts_for_entry = None
                image_metadata = _extract_image_metadata_from_entry(entry)
                if image_metadata and len(image_metadata) == len(processed_images):
                    image_prompts_for_entry = []
                    for meta in image_metadata:
                        if "vlm_prompt" in meta:
                            image_prompts_for_entry.append(meta["vlm_prompt"])
                        else:
                            image_prompts_for_entry = None
                            break

                if auth_abort.is_set():
                    raise ModelAuthenticationFailure()
                response = await async_classify_object(
                    vlm=vlm,
                    text=text,
                    images=processed_images,
                    llm=llm,
                    system_prompt=entry_system_prompt,
                    invoke_kwargs=invoke_kwargs,
                    image_prompts=image_prompts_for_entry,
                    max_retries=max_retries,
                    output_key=output_key,
                    token_tracker=token_tracker,
                    unknown_sentinel=unknown_sentinel,
                )

                # Log response preview
                if isinstance(response, dict):
                    response_preview = f"{output_key}='{response.get(output_key)}'"
                else:
                    response_preview = response[:100] if response else "No response"
                logger.info(f"Entry {entry_id} response: {response_preview}")

                # Timing (lock protects concurrent stat updates and reads)
                elapsed_seconds = perf_counter() - start_time
                async with stats_lock:
                    total_assignment_seconds += elapsed_seconds
                    timed_assignments += 1
                    avg_seconds = total_assignment_seconds / max(timed_assignments, 1)
                    remaining = max(len(entries_to_process) - timed_assignments, 0)
                    eta_seconds = avg_seconds * remaining
                    completed = timed_assignments
                finish_time_local = datetime.now() + timedelta(seconds=int(eta_seconds))
                logger.info(
                    "Timing: last=%ss, avg=%ss, completed=%d/%d, remaining≈%s, ETA %s",
                    f"{elapsed_seconds:.1f}",
                    f"{avg_seconds:.1f}",
                    completed,
                    len(entries_to_process),
                    _format_duration(eta_seconds),
                    finish_time_local.strftime("%Y-%m-%d %H:%M:%S"),
                )

                result: dict[str, Any] = {
                    "id": entry_id,
                    "vlm_response": response,
                    "status": "success",
                }

                if on_progress:
                    on_progress(entry_id, response)
                if on_prediction:
                    try:
                        on_prediction(entry_id, response)
                    except Exception as cb_e:
                        logger.warning(
                            f"on_prediction callback failed for {entry_id}: {cb_e}"
                        )
                if on_result:
                    try:
                        on_result(result, entry)
                    except Exception as cb_e:
                        logger.warning(
                            f"on_result callback failed for {entry_id}: {cb_e}"
                        )

                return result

            except Exception as e:
                try:
                    raise_for_model_authentication(e)
                except ModelAuthenticationFailure:
                    auth_abort.set()
                    raise
                error_msg = str(e)
                logger.error(f"Entry {entry_id} failed: {error_msg}", exc_info=True)
                if on_error:
                    on_error(entry_id, error_msg)

                error_result = {
                    "id": entry_id,
                    "vlm_response": None,
                    "status": "error",
                    "error": error_msg,
                }
                if on_result:
                    try:
                        on_result(error_result, entry)
                    except Exception as cb_e:
                        logger.warning(
                            f"on_result callback failed for {entry_id}: {cb_e}"
                        )
                return error_result

    # Launch all entries with gather (semaphore controls concurrency)
    tasks = [
        asyncio.create_task(_process_entry(entry, idx))
        for idx, entry in enumerate(entries_to_process, 1)
    ]
    try:
        results = await asyncio.gather(*tasks)
    except Exception as error:
        try:
            raise_for_model_authentication(error)
        except ModelAuthenticationFailure as auth_failure:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise auth_failure from None
        raise

    # Log summary
    results_list = list(results)
    successful = sum(1 for r in results_list if r["status"] == "success")
    failed = sum(1 for r in results_list if r["status"] == "error")
    logger.info(
        f"Async batch classification complete: {successful} successful, "
        f"{failed} failed out of {len(entries_to_process)} total "
        f"(concurrency={max_workers or 1})"
    )

    return results_list
