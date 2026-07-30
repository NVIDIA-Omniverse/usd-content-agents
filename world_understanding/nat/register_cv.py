# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
from typing import Any

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class DominantColorsToolConfig(FunctionBaseConfig, name="get_dominant_colors"):  # type: ignore[call-arg]
    pass


@register_function(config_type=DominantColorsToolConfig)  # type: ignore[misc]
async def get_dominant_colors(
    config: DominantColorsToolConfig, builder: Builder
) -> Any:
    import json

    from world_understanding.functions.cv.get_dominant_colors import (
        get_dominant_colors as get_dominant_colors_fn,
    )

    async def _get_dominant_colors(image_path: str, n_colors: int = 5) -> str:
        """
        Extract dominant colors from an image.

        Args:
            image_path: Path to the image file to analyze
            n_colors: Number of dominant colors to extract (1-20)

        Returns:
            JSON string with dominant colors and analysis
        """
        try:
            # Call the portable function
            result = await asyncio.to_thread(
                get_dominant_colors_fn,
                image=image_path,
                n_colors=n_colors,
                analyze_brightness=True,
            )

            # Format the result for better readability
            formatted_result = {
                "dominant_colors": [
                    {
                        "hex": color["hex"],
                        "rgb": color["rgb"],
                        "percentage": f"{color['percentage'] * 100:.1f}%",
                    }
                    for color in result["dominant_colors"]
                ],
                "average_brightness": (f"{result['average_brightness']:.1f}/255"),
                "color_diversity": f"{result['color_diversity']:.3f}",
                "n_clusters": result["n_clusters"],
            }

            return json.dumps(formatted_result, indent=2)

        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except Exception as e:
            return f"Failed to analyze image: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _get_dominant_colors,
        description=(
            "Extract dominant colors from an image using k-means "
            "clustering. This tool analyzes an image and returns the "
            "most prominent colors, their percentages, average "
            "brightness, and color diversity metrics."
        ),
    )


class FindSimilarColorToolConfig(FunctionBaseConfig, name="find_similar_color"):  # type: ignore[call-arg]
    pass


@register_function(config_type=FindSimilarColorToolConfig)  # type: ignore[misc]
async def find_similar_color(
    config: FindSimilarColorToolConfig, builder: Builder
) -> Any:
    import json

    from world_understanding.functions.cv.find_similar_color import (
        find_similar_color as find_similar_color_fn,
    )

    async def _find_similar_color(
        image_path: str,
        target_r: int,
        target_g: int,
        target_b: int,
        tolerance: int = 50,
    ) -> str:
        """
        Check if an image contains a specific RGB color.

        Args:
            image_path: Path to the image file
            target_r: Red component (0-255)
            target_g: Green component (0-255)
            target_b: Blue component (0-255)
            tolerance: Color matching tolerance (0-255)

        Returns:
            JSON string with color matching results
        """
        try:
            # Call the portable function
            result = await asyncio.to_thread(
                find_similar_color_fn,
                image=image_path,
                target_color=[target_r, target_g, target_b],
                color_tolerance=tolerance,
                min_percentage=1.0,
            )

            # Format the result
            formatted_result = {
                "contains_color": result["contains_color"],
                "matching_percentage": f"{result['matching_percentage']:.2f}%",
                "pixel_count": result["pixel_count"],
                "total_pixels": result["total_pixels"],
                "target_color": {
                    "rgb": result["target_color_rgb"],
                    "hex": result["target_color_hex"],
                },
            }

            # Add closest colors if available
            if "closest_colors" in result and result["closest_colors"]:
                formatted_result["closest_colors"] = [
                    {
                        "hex": color["hex"],
                        "rgb": color["rgb"],
                        "distance": f"{color['distance']:.1f}",
                    }
                    for color in result["closest_colors"][:3]
                ]

            return json.dumps(formatted_result, indent=2)

        except ValueError as e:
            return f"Invalid parameter: {str(e)}"
        except FileNotFoundError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Failed to match color: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _find_similar_color,
        description=(
            "Check if an image contains a specific RGB color within "
            "a tolerance range. Useful for verifying if certain colors "
            "are present in images, finding objects by color, or "
            "validating color requirements."
        ),
    )


class VLMToolConfig(FunctionBaseConfig, name="vlm"):  # type: ignore[call-arg]
    pass


@register_function(config_type=VLMToolConfig)  # type: ignore[misc]
async def vlm(config: VLMToolConfig, builder: Builder) -> Any:
    import os

    from PIL import Image

    from world_understanding.functions.cv.vlm import generate_vlm_response
    from world_understanding.functions.models.vision_language_models import create_vlm

    async def _vlm(
        image_path: str,
        prompt: str,
        backend: str = "nim",
        model: str | None = None,
        system_prompt: str = "You are a helpful AI assistant that can analyze images.",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Analyze an image using Vision-Language Models.

        Args:
            image_path: Path to the image file to analyze
            prompt: Question or prompt about the image
            backend: VLM backend to use (e.g. nim, openai, anthropic, gemini)
            model: Model to use (optional, uses backend default)
            system_prompt: System instructions for the model
            temperature: Temperature for response generation (0.0-2.0)
            max_tokens: Maximum tokens in response

        Returns:
            Model's text response about the image
        """
        try:
            # Load image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resolve a default NIM model if the caller didn't pick one.
            # For other backends, create_vlm picks the right model/key from env.
            api_key = None
            if backend == "nim":
                api_key = os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    return "Error: NVIDIA_API_KEY environment variable not set"
                if not model:
                    model = "google/gemma-4-31b-it"

            # Create VLM instance using the new create_vlm function
            try:
                vlm = create_vlm(
                    backend=backend,
                    api_key=api_key,
                    model=model,
                )
            except Exception as e:
                return f"Error creating VLM: {str(e)}"

            # Call the portable function
            result = await asyncio.to_thread(
                generate_vlm_response,
                vlm=vlm,
                prompt=prompt,
                system_prompt=system_prompt,
                images=[image],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if "error" in result:
                return f"Error: {result['error']}"

            return result["response"]

        except FileNotFoundError:
            return f"Error: Image file not found: {image_path}"
        except Exception as e:
            return f"Failed to analyze image: {str(e)}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(
        _vlm,
        description=(
            "Analyze images using Vision-Language Models to answer questions "
            "about visual content. Supports multiple VLM services including "
            "NVIDIA NIM and Azure OpenAI with various state-of-the-art models."
        ),
    )
