# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for retrieving materials using the USD Search API."""

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from world_understanding.agentic.events import EventListener, get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.functions.knowledge.usd_search import USDSearchClient
from world_understanding.functions.models.chat_models import create_chat_model

from material_agent.materials import (
    FALLBACK_MATERIAL_BINDING,
    FALLBACK_MATERIAL_NAME,
    UNKNOWN_MATERIAL_SENTINEL,
    is_actionable_material_name,
    is_unknown_material_name,
    material_mapping_with_fallback,
)

logger = logging.getLogger(__name__)


class MaterialRetrievalTask(Task):
    """Task to retrieve materials using the USD Search API or direct mapping.

    This task takes a list of unique materials and retrieves matching materials
    using either a direct materials_mapping (if provided) or the USD Search API.
    It returns the search results including material paths found for each material.

    Input context keys:
        - unique_materials: List of unique material names to search for
        - materials_mapping: Optional dict mapping material names to file paths
                           Also supports legacy list[dict] format for backward compatibility
        - usd_search_config: Configuration for USD Search (limit, host, etc.)

    Output context keys:
        - matched_materials: Dictionary mapping material names to lists of found paths
        - search_results: Raw search results from USD Search API or direct mapping
        - search_stats: Statistics about the search (total queries, matches, etc.)
        - unresolved_materials: List of material names that could not be resolved
    """

    def __init__(self):
        """Initialize the material retrieval task."""
        self.name = "MaterialRetrieval"
        self.description = "Retrieve materials using USD Search API"

    def _use_materials_mapping(
        self,
        context: dict[str, Any],
        unique_materials: list[str],
        materials_mapping: dict[str, str] | list[dict[str, str]],
        listener,
    ) -> dict[str, Any]:
        """Use direct materials mapping instead of USD Search.

        Args:
            context: Workflow context
            unique_materials: List of unique material names
            materials_mapping: Dictionary mapping material names to paths, or
                             list of dicts for backward compatibility
            listener: Event listener for progress reporting

        Returns:
            Updated context with mapped materials
        """
        # Handle both dict and list[dict] formats
        if isinstance(materials_mapping, dict):
            # Direct dictionary format (preferred)
            mapping_dict = dict(materials_mapping)
        else:
            # Legacy list of dicts format - convert to dictionary
            mapping_dict = {}
            for item in materials_mapping:
                if isinstance(item, dict):
                    mapping_dict.update(item)
        mapping_dict = material_mapping_with_fallback(mapping_dict)

        # Check if this is a library-based mapping
        material_library_path = mapping_dict.get("material_library_path")
        if material_library_path:
            listener.info(
                f"Detected library-based material mapping using: {material_library_path}"
            )
            return self._use_library_based_mapping(
                context, unique_materials, mapping_dict, material_library_path, listener
            )

        matched_materials = {}
        search_results = {}
        found_count = 0
        missing_count = 0
        unresolved_materials = []

        for material in unique_materials:
            if material in mapping_dict:
                s3_uri = mapping_dict[material]

                # Parse S3 URI to extract path and bucket
                # Supports both formats:
                #   - s3://bucket-name/path/to/file.mdl
                #   - /path/to/file.mdl (without s3:// prefix)
                source_path = s3_uri
                s3_path = s3_uri

                if s3_uri.startswith("s3://"):
                    # Extract bucket and path from S3 URI
                    # Format: s3://bucket-name/path/to/file
                    uri_parts = s3_uri[5:]  # Remove 's3://'
                    if "/" in uri_parts:
                        bucket_name, path = uri_parts.split("/", 1)
                        source_path = "/" + path  # source_path should start with /
                        s3_path = s3_uri  # Keep full S3 URI
                        listener.debug(
                            f"Parsed S3 URI: bucket={bucket_name}, path={source_path}"
                        )

                # Create path info in the expected format
                path_info = {
                    "source_path": source_path,
                    "s3_path": s3_path,
                    "dependencies": [],
                    "metadata": {"source": "materials_mapping"},
                }
                matched_materials[material] = [path_info]
                search_results[material] = [{"source": {"path": source_path}}]
                found_count += 1
                listener.info(f"Mapped '{material}' -> {s3_uri}")
            else:
                listener.warning(
                    f"Material '{material}' not found in materials_mapping"
                )
                matched_materials[material] = []
                search_results[material] = []
                unresolved_materials.append(material)
                missing_count += 1

        # Update context with results
        context["matched_materials"] = matched_materials
        context["search_results"] = search_results
        context["unresolved_materials"] = unresolved_materials
        context["search_stats"] = {
            "total_queries": len(unique_materials),
            "total_matches": found_count,
            "failed_queries": missing_count,
        }

        listener.info(
            f"Materials mapping complete: {found_count} found, {missing_count} missing"
        )

        # Report unresolved materials prominently if any exist
        if unresolved_materials:
            listener.warning(
                f"\n{'=' * 80}\n"
                f"UNRESOLVED MATERIALS: {len(unresolved_materials)} material(s) not found in materials_mapping\n"
                f"{'=' * 80}"
            )
            for material in unresolved_materials:
                listener.warning(f"  - {material}")
            listener.warning(f"{'=' * 80}\n")

        return context

    def _use_library_based_mapping(
        self,
        context: dict[str, Any],
        unique_materials: list[str],
        mapping_dict: dict[str, str],
        material_library_path: str,
        listener,
    ) -> dict[str, Any]:
        """Use USD material library-based mapping.

        Args:
            context: Workflow context
            unique_materials: List of unique material names
            mapping_dict: Dictionary with material_library_path and material->prim path mappings
            material_library_path: Path to the USD material library file
            listener: Event listener for progress reporting

        Returns:
            Updated context with library-based mapped materials
        """
        from pathlib import Path

        # Validate library path exists
        library_path = Path(material_library_path)
        if not library_path.exists():
            listener.error(f"Material library file not found: {material_library_path}")
            raise FileNotFoundError(
                f"Material library file not found: {material_library_path}"
            )

        matched_materials = {}
        search_results = {}
        found_count = 0
        missing_count = 0
        unresolved_materials = []

        # Process each material
        for material in unique_materials:
            if material in mapping_dict and material != "material_library_path":
                prim_path = mapping_dict[material]

                # Create path info with library metadata
                path_info = {
                    "source_path": prim_path,  # Prim path within library
                    "s3_path": None,  # Not applicable for library materials
                    "dependencies": [],
                    "metadata": {
                        "source": "materials_library",
                        "library_path": material_library_path,
                        "is_library_material": True,
                    },
                }
                matched_materials[material] = [path_info]
                search_results[material] = [
                    {
                        "source": {
                            "path": prim_path,
                            "library": material_library_path,
                        }
                    }
                ]
                found_count += 1
                listener.info(
                    f"Mapped '{material}' -> library prim: {prim_path} (from {library_path.name})"
                )
            else:
                listener.warning(
                    f"Material '{material}' not found in library-based materials_mapping"
                )
                matched_materials[material] = []
                search_results[material] = []
                unresolved_materials.append(material)
                missing_count += 1

        # Store library path in context for downstream tasks
        context["material_library_path"] = material_library_path
        context["is_library_based_mapping"] = True

        # Update context with results
        context["matched_materials"] = matched_materials
        context["search_results"] = search_results
        context["unresolved_materials"] = unresolved_materials
        context["search_stats"] = {
            "total_queries": len(unique_materials),
            "total_matches": found_count,
            "failed_queries": missing_count,
        }

        listener.info(
            f"Library-based mapping complete: {found_count} found, {missing_count} missing"
        )

        # Report unresolved materials prominently if any exist
        if unresolved_materials:
            listener.warning(
                f"\n{'=' * 80}\n"
                f"UNRESOLVED MATERIALS: {len(unresolved_materials)} material(s) not found in library mapping\n"
                f"{'=' * 80}"
            )
            for material in unresolved_materials:
                listener.warning(f"  - {material}")
            listener.warning(f"{'=' * 80}\n")

        return context

    def run(self, context: dict[str, Any], object_store=None) -> dict[str, Any]:
        """Retrieve materials using USD Search or direct mapping.

        Args:
            context: Workflow context containing unique_materials
            object_store: Optional object store (not used)

        Returns:
            Updated context with search results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        unique_materials = context.get("unique_materials", [])
        unknown_materials = [m for m in unique_materials if is_unknown_material_name(m)]
        normalized_materials = []
        for material in unique_materials:
            if is_unknown_material_name(material):
                normalized_materials.append(FALLBACK_MATERIAL_NAME)
            elif is_actionable_material_name(material):
                normalized_materials.append(material)
        unique_materials = sorted(set(normalized_materials))
        if unknown_materials:
            listener.warning(
                f"Ignoring {len(unknown_materials)} "
                f"'{UNKNOWN_MATERIAL_SENTINEL}' material sentinel(s); "
                f"retrieving '{FALLBACK_MATERIAL_NAME}' instead"
            )
            context["unique_materials"] = unique_materials
            existing_unknown_count = context.get("unknown_material_predictions", 0)
            if not isinstance(existing_unknown_count, int):
                existing_unknown_count = 0
            context["unknown_material_predictions"] = max(
                existing_unknown_count,
                len(unknown_materials),
            )
        if not unique_materials:
            listener.warning("No unique materials found to retrieve")
            context["matched_materials"] = {}
            context["search_results"] = {}
            context["unresolved_materials"] = []
            context["search_stats"] = {
                "total_queries": 0,
                "total_matches": 0,
                "failed_queries": 0,
            }
            return context

        # Emit task started event
        listener.event(
            "task.started",
            {
                "task_name": "MaterialRetrieval",
                "total_materials": len(unique_materials),
                "materials": unique_materials,
            },
        )

        # Check if materials_mapping is provided for direct mapping
        materials_mapping = context.get("materials_mapping")
        if materials_mapping:
            listener.info(
                f"Using direct materials_mapping for {len(unique_materials)} materials"
            )
            return self._use_materials_mapping(
                context, unique_materials, materials_mapping, listener
            )

        # Get USD Search configuration
        usd_search_config = context.get("usd_search_config", {})
        limit = usd_search_config.get("limit", 10)
        host = usd_search_config.get("host")  # None uses default
        file_extensions = usd_search_config.get("file_extension_include", ["mdl"])

        # Get LLM configuration for enhanced retrieval (optional)
        llm_config = context.get("llm_config")
        use_llm_retrieval = llm_config is not None and usd_search_config.get(
            "use_llm_enhanced_search", False
        )

        listener.info(f"Retrieving {len(unique_materials)} materials using USD Search")
        listener.info(
            f"Retrieval configuration: limit={limit}, host={host or 'default'}, "
            f"file_extensions={file_extensions}"
        )
        if use_llm_retrieval:
            listener.info("LLM-enhanced hierarchical retrieval enabled")

        # Initialize USD Search client
        client = USDSearchClient(host=host)

        # Initialize LLM if configured for enhanced retrieval
        llm = None
        if use_llm_retrieval:
            try:
                llm = self._create_llm(llm_config)
                listener.info("LLM initialized for enhanced material retrieval")
            except Exception as e:
                listener.warning(
                    f"Failed to initialize LLM: {e}. Falling back to standard retrieval."
                )
                use_llm_retrieval = False

        # Retrieve each material
        matched_materials = {}
        search_results = {}
        search_materials = []
        for material in unique_materials:
            if material == FALLBACK_MATERIAL_NAME:
                path_info = {
                    "source_path": FALLBACK_MATERIAL_BINDING,
                    "s3_path": None,
                    "dependencies": [],
                    "metadata": {"source": "canonical_fallback"},
                }
                matched_materials[material] = [path_info]
                search_results[material] = [
                    {"source": {"path": FALLBACK_MATERIAL_BINDING}}
                ]
                listener.info(f"Mapped '{material}' -> canonical fallback material")
            else:
                search_materials.append(material)
        failed_queries = 0

        for material in search_materials:
            try:
                if use_llm_retrieval and llm:
                    # Use LLM-enhanced hierarchical retrieval
                    listener.info(f"Retrieving material: '{material}' (LLM-enhanced)")
                    paths, results = self._llm_enhanced_retrieval(
                        material, client, llm, limit, file_extensions, listener
                    )
                else:
                    # Use standard retrieval
                    listener.info(f"Retrieving material: '{material}' (standard)")
                    results = client.search(
                        query=material,
                        limit=limit,
                        return_metadata=True,
                        return_images=False,
                        file_extension_include=file_extensions,
                    )
                    paths = self._extract_paths_from_results(results, listener)

                matched_materials[material] = paths
                search_results[material] = results

                listener.info(f"Found {len(paths)} matches for '{material}'")

                if logger.isEnabledFor(logging.DEBUG) and paths:
                    listener.debug(f"  Sample paths: {paths[:3]}")  # Show first 3 paths

            except Exception as e:
                listener.error(f"Failed to retrieve material '{material}': {e}")
                matched_materials[material] = []
                search_results[material] = []
                failed_queries += 1

        # Calculate statistics and identify unresolved materials
        total_matches = sum(len(paths) for paths in matched_materials.values())
        unresolved_materials = [
            material for material, paths in matched_materials.items() if not paths
        ]

        retrieval_stats = {
            "total_queries": len(unique_materials),
            "total_matches": total_matches,
            "failed_queries": failed_queries,
            "successful_queries": len(unique_materials) - failed_queries,
        }

        listener.info(
            f"Material retrieval completed: {retrieval_stats['successful_queries']}/{retrieval_stats['total_queries']} successful queries, {total_matches} total matches"
        )

        # Update context
        context["matched_materials"] = matched_materials
        context["search_results"] = search_results
        context["search_stats"] = retrieval_stats
        context["unresolved_materials"] = unresolved_materials

        # Report unresolved materials prominently if any exist
        if unresolved_materials:
            listener.warning(
                f"\n{'=' * 80}\n"
                f"UNRESOLVED MATERIALS: {len(unresolved_materials)} material(s) could not be found\n"
                f"{'=' * 80}"
            )
            for material in unresolved_materials:
                listener.warning(f"  - {material}")
            listener.warning(f"{'=' * 80}\n")

        # Log material retrieval results for user visibility
        self._log_retrieval_summary(matched_materials, listener)

        # Emit task completed event
        total_matches = sum(len(paths) for paths in matched_materials.values())
        listener.event(
            "task.completed",
            {
                "task_name": "MaterialRetrieval",
                "total_materials": len(unique_materials),
                "matched_materials": len(matched_materials),
                "total_matches": total_matches,
                "unresolved": len(context.get("unresolved_materials", [])),
            },
        )

        return context

    def _create_llm(self, llm_config: dict[str, Any]) -> Any:
        """Create an LLM instance from configuration.

        Args:
            llm_config: LLM configuration dictionary

        Returns:
            LLM instance
        """
        service = llm_config.get("service", "nim")
        model = llm_config.get("model")
        temperature = llm_config.get("temperature", 0.1)
        max_tokens = llm_config.get("max_tokens", 512)

        # Get API key from config or environment
        api_key = llm_config.get("api_key")
        if not api_key:
            if service == "nim":
                api_key = os.getenv("NVIDIA_API_KEY")

        return create_chat_model(
            backend=service,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _parse_material_with_llm(
        self, material_name: str, llm: Any, listener: Any
    ) -> dict[str, str]:
        """Parse a material name into components using LLM.

        Args:
            material_name: Full material name to parse
            llm: LLM instance
            listener: Event listener for progress reporting

        Returns:
            Dictionary with 'material', 'color', and 'finish' keys
        """
        system_prompt = """You are a material classification expert. Parse material names into their components.
Extract the functional material type, color, and finish from material names.

Important:
- Material (functional material): The base material type (e.g., aluminum, steel, plastic, rubber, glass, wood)
- Color: The color descriptor (e.g., black, silver, red, transparent). Use "none" if not specified.
- Finish: The surface finish (e.g., matte, glossy, brushed, textured). Use "none" if not specified.

Return ONLY a valid JSON object with this exact format:
{"material": "base_material", "color": "color", "finish": "finish"}

DO NOT explain your response, just return the JSON object."""

        user_prompt = f"""Parse this material name: "{material_name}"

Return the JSON object."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = llm.invoke(messages)
            response_text = response.content

            listener.debug(f"LLM parse request for '{material_name}':")
            listener.debug(f"  System: {system_prompt[:100]}...")
            listener.debug(f"  User: {user_prompt}")
            listener.debug(f"LLM parse response: {response_text}")

            # Extract JSON from response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)

                # Validate required keys
                if "material" in parsed:
                    result = {
                        "material": parsed.get("material", "").strip(),
                        "color": parsed.get("color", "none").strip(),
                        "finish": parsed.get("finish", "none").strip(),
                    }
                    listener.info(
                        f"Parsed '{material_name}' -> material: '{result['material']}', "
                        f"color: '{result['color']}', finish: '{result['finish']}'"
                    )
                    return result

            # Fallback: use the original material name as material
            listener.warning(f"Failed to parse '{material_name}', using as-is")
            listener.debug(f"Unparseable LLM response: {response_text}")
            return {"material": material_name, "color": "none", "finish": "none"}

        except Exception as e:
            listener.error(f"Error parsing material '{material_name}': {e}")
            if "response_text" in locals():
                listener.debug(f"LLM response that caused error: {response_text}")
            return {"material": material_name, "color": "none", "finish": "none"}

    def _select_best_match_with_llm(
        self,
        original_material: str,
        parsed_material: dict[str, str],
        search_results: list[dict[str, Any]],
        llm: Any,
        listener: Any,
    ) -> int | None:
        """Use LLM to select the best matching material from search results.

        Args:
            original_material: Original material name
            parsed_material: Parsed material components (material, color, finish)
            search_results: List of search results from USD Search
            llm: LLM instance
            listener: Event listener for progress reporting

        Returns:
            Index of the best match in search_results, or None if no good match
        """
        if not search_results:
            return None

        # Extract material names/paths from search results for LLM
        candidates = []
        for i, result in enumerate(search_results):
            # Try to extract a meaningful name from the result
            name = None
            if "source" in result and isinstance(result["source"], dict):
                name = result["source"].get("path", result["source"].get("base_key"))
            if not name:
                name = result.get("id", f"candidate_{i}")

            candidates.append(f"{i}: {name}")

        candidates_text = "\n".join(candidates)

        system_prompt = """You are a material matching expert. Given a target material description and a list of candidate materials, select the best match.

Consider:
1. Material type (most important) - must match the functional material
2. Color (secondary) - should match if specified
3. Finish (tertiary) - should match if specified

Return ONLY a valid JSON object with this format:
{"best_match_index": <index>, "reasoning": "brief explanation"}

If no good match exists, return: {"best_match_index": null, "reasoning": "explanation"}"""

        user_prompt = f"""Target material: "{original_material}"
Parsed components:
- Material: {parsed_material["material"]}
- Color: {parsed_material["color"]}
- Finish: {parsed_material["finish"]}

Candidate materials:
{candidates_text}

Select the best match (return the index) or null if no good match.

DO NOT use automotive materials unless explicitly specified."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            response = llm.invoke(messages)
            response_text = response.content

            listener.debug(f"LLM match selection request for '{original_material}':")
            listener.debug(f"  System: {system_prompt[:100]}...")
            listener.debug(f"  User prompt:\n{user_prompt}")
            listener.debug(f"LLM match selection response:\n{response_text}")

            # Extract JSON from response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)

                best_match_index = result.get("best_match_index")
                reasoning = result.get("reasoning", "")

                if best_match_index is not None:
                    listener.info(
                        f"LLM selected candidate {best_match_index} for '{original_material}': {reasoning}"
                    )
                else:
                    listener.info(
                        f"LLM found no good match for '{original_material}': {reasoning}"
                    )

                return best_match_index

            # Fallback: return first result
            listener.warning("Failed to parse LLM response, using first result")
            listener.debug(f"Unparseable LLM response: {response_text}")
            return 0

        except Exception as e:
            listener.error(f"Error in LLM match selection: {e}")
            if "response_text" in locals():
                listener.debug(f"LLM response that caused error: {response_text}")
            return 0

    def _llm_enhanced_retrieval(
        self,
        material_name: str,
        client: USDSearchClient,
        llm: Any,
        limit: int,
        file_extensions: list[str],
        listener: Any,
    ) -> tuple[list[dict[str, str | None]], list[dict[str, Any]]]:
        """Perform LLM-enhanced hierarchical material retrieval.

        This implements a two-stage retrieval:
        1. Parse material into (material, color, finish) using LLM
        2. Search USD with just the material part
        3. Use LLM to select best match considering full color and finish

        Args:
            material_name: Material name to search for
            client: USD Search client
            llm: LLM instance
            limit: Maximum number of results to return
            file_extensions: File extensions to filter
            listener: Event listener for progress reporting

        Returns:
            Tuple of (paths, results) - paths is list of path dicts, results is list of raw results
        """
        # Step 1: Parse material name into components
        parsed = self._parse_material_with_llm(material_name, llm, listener)

        # Step 2: Search with just the functional material part
        # Use configured limit with minimum of 10
        search_limit = max(limit, 10)
        listener.info(
            f"Searching for base material '{parsed['material']}' (limit={search_limit})"
        )

        results = client.search(
            query=parsed["material"],
            limit=search_limit,
            return_metadata=True,
            return_images=False,
            file_extension_include=file_extensions,
        )

        if not results:
            listener.warning(
                f"No results found for base material '{parsed['material']}'"
            )
            return [], []

        listener.info(
            f"Found {len(results)} candidate materials for '{parsed['material']}'"
        )

        # Step 3: Use LLM to select the best match(es) considering color and finish
        best_match_idx = self._select_best_match_with_llm(
            material_name, parsed, results, llm, listener
        )

        if best_match_idx is None:
            listener.warning(
                f"LLM found no suitable match for '{material_name}', "
                f"falling back to first candidate from USD Search"
            )
            # Fall back to first candidate when LLM finds no suitable match
            best_match_idx = 0

        # Return only the best match (or first candidate as fallback)
        selected_results = [results[best_match_idx]]
        paths = self._extract_paths_from_results(selected_results, listener)

        return paths, selected_results

    def _extract_paths_from_results(
        self, results: list[dict[str, Any]], listener: Any
    ) -> list[dict[str, str | None]]:
        """Extract file paths from USD Search results.

        Args:
            results: List of search result dictionaries
            listener: Event listener for progress reporting

        Returns:
            List of dictionaries containing source_path and s3_path
        """
        paths = []

        for result in results:
            try:
                path_info = self._extract_path_from_result(result, listener)
                if path_info and (
                    path_info.get("source_path") or path_info.get("s3_path")
                ):
                    paths.append(path_info)
            except Exception as e:
                listener.debug(f"Failed to extract path from result: {e}")
                continue

        return paths

    def _extract_path_from_result(
        self, result: dict[str, Any], listener: Any
    ) -> dict[str, str | None]:
        """Extract file paths from a single search result.

        Args:
            result: Single search result dictionary
            listener: Event listener for progress reporting

        Returns:
            Dictionary containing source_path, s3_path, and dependencies
        """
        path_info = {
            "source_path": None,
            "s3_path": None,
            "dependencies": [],
            "metadata": {},
        }

        # Extract both paths from source if available
        if "source" in result and isinstance(result["source"], dict):
            # Get S3 path from base_key
            if "base_key" in result["source"] and isinstance(
                result["source"]["base_key"], str
            ):
                path_info["s3_path"] = result["source"]["base_key"]

            # Get source path from path
            if "path" in result["source"] and isinstance(result["source"]["path"], str):
                path_info["source_path"] = result["source"]["path"]

        # Extract metadata for inspection
        if "metadata" in result and isinstance(result["metadata"], dict):
            metadata = result["metadata"]
            path_info["metadata"] = metadata

            # Log full metadata structure for debugging (first result only)
            if logger.isEnabledFor(logging.DEBUG):
                import json

                listener.debug(
                    f"Full metadata structure: {json.dumps(metadata, indent=2, default=str)}"
                )

            # Look for dependency information in various possible fields
            dependency_fields = [
                "dependencies",
                "resources",
                "related_files",
                "textures",
                "linked_files",
                "assets",
                "auxiliary_files",
            ]

            for field in dependency_fields:
                if field in metadata:
                    deps = metadata[field]
                    if isinstance(deps, list):
                        path_info["dependencies"].extend(deps)
                        listener.info(
                            f"Found {len(deps)} dependencies in '{field}': {deps}"
                        )
                    elif isinstance(deps, dict):
                        # Dependencies might be a dict mapping names to paths
                        dep_list = list(deps.values()) if deps else []
                        path_info["dependencies"].extend(dep_list)
                        listener.info(
                            f"Found {len(dep_list)} dependencies in '{field}' dict"
                        )
                    elif isinstance(deps, str):
                        # Single dependency as string
                        path_info["dependencies"].append(deps)
                        listener.info(f"Found single dependency in '{field}': {deps}")

        # If no source paths found, try to find a path from other fields
        if not path_info["source_path"] and not path_info["s3_path"]:
            # Common field names that might contain paths
            path_fields = [
                "path",
                "file_path",
                "filepath",
                "url",
                "uri",
                "location",
                "id",  # Sometimes the ID is the path
            ]

            # Check direct fields
            for field in path_fields:
                if field in result and isinstance(result[field], str):
                    path_info["source_path"] = result[field]
                    break

            # Check nested metadata
            if (
                not path_info["source_path"]
                and "metadata" in result
                and isinstance(result["metadata"], dict)
            ):
                for field in path_fields:
                    if field in result["metadata"] and isinstance(
                        result["metadata"][field], str
                    ):
                        path_info["source_path"] = result["metadata"][field]
                        break

            # Check nested in other common structures
            if not path_info["source_path"]:
                for nested_key in ["data", "document", "item"]:
                    if nested_key in result and isinstance(result[nested_key], dict):
                        for field in path_fields:
                            if field in result[nested_key] and isinstance(
                                result[nested_key][field], str
                            ):
                                path_info["source_path"] = result[nested_key][field]
                                break
                        if path_info["source_path"]:
                            break

        # If nothing found, log the structure for debugging
        if not path_info["source_path"] and not path_info["s3_path"]:
            listener.debug(
                f"Could not extract paths from result with keys: {result.keys()}"
            )

        return path_info

    def _log_retrieval_summary(
        self,
        matched_materials: dict[str, list[dict[str, str | None]]],
        listener: EventListener,
    ) -> None:
        """Log a summary of the retrieval results.

        Args:
            matched_materials: Dictionary mapping materials to found paths
            listener: Event listener for logging
        """
        listener.info("=" * 50)
        listener.info("MATERIAL RETRIEVAL RESULTS SUMMARY")
        listener.info("=" * 50)

        if not matched_materials:
            listener.info("No materials were retrieved.")
            return

        for material, path_infos in matched_materials.items():
            listener.info(f"Material: '{material}' - Matches found: {len(path_infos)}")

            if path_infos:
                for i, path_info in enumerate(path_infos[:5]):  # Show first 5 paths
                    source = path_info.get("source_path", "")
                    s3 = path_info.get("s3_path", "")
                    listener.debug(f"  {i + 1}. Source: {source}  S3: {s3}")
                if len(path_infos) > 5:
                    listener.debug(f"  ... and {len(path_infos) - 5} more matches")
            else:
                listener.info(f"  No matching paths found for '{material}'")

        listener.info("=" * 50)
