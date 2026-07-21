# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene Optimizer path correspondence helpers for Content Workbench."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _natural_sort_key(text: str) -> list[tuple[int, int | str]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part)
        for part in re.split(r"(\d+)", text)
    ]


@dataclass(frozen=True)
class PathTranslation:
    """Result of translating one prim path between scene coordinate spaces."""

    input_path: str
    source_paths: list[str]
    inspection_paths: list[str]
    ambiguous: bool = False


@dataclass
class SceneOptimizerPathMap:
    """Bidirectional source/inspection prim path map from SO metadata."""

    source_to_inspection_map: dict[str, list[str]] = field(default_factory=dict)
    inspection_to_source_map: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_metadata(
        cls,
        *,
        original_usd_path: Path,
        optimization_metadata: dict[str, Any],
    ) -> SceneOptimizerPathMap:
        """Build a path map from ``OptimizeUSDTask`` metadata."""
        correspondence_map = optimization_metadata.get("correspondence_map", {})
        if not isinstance(correspondence_map, dict):
            return cls()

        full_mapping = correspondence_map.get("full_mapping", {})
        if not isinstance(full_mapping, dict):
            return cls()

        original_to_prototype = full_mapping.get("original_to_prototype", {})
        if not isinstance(original_to_prototype, dict):
            return cls()

        split_mapping = correspondence_map.get("split_mapping", {})
        if not isinstance(split_mapping, dict):
            split_mapping = {}

        subset_paths = _geomsubset_paths_by_parent(original_usd_path)
        source_to_inspection: dict[str, list[str]] = defaultdict(list)
        inspection_to_source: dict[str, list[str]] = defaultdict(list)

        for original_path, raw_prototypes in original_to_prototype.items():
            if not isinstance(original_path, str):
                continue
            prototypes = sorted(
                _coerce_path_list(raw_prototypes),
                key=_natural_sort_key,
            )
            if not prototypes:
                continue

            source_to_inspection[original_path].extend(prototypes)
            if original_path in split_mapping:
                subsets = sorted(
                    subset_paths.get(original_path, []),
                    key=_natural_sort_key,
                )
                for index, prototype_path in enumerate(prototypes):
                    source_path = (
                        subsets[index]
                        if index < len(subsets)
                        else f"{original_path}_part_{index}"
                    )
                    source_to_inspection[source_path].append(prototype_path)
                    inspection_to_source[prototype_path].append(source_path)
                continue

            for prototype_path in prototypes:
                inspection_to_source[prototype_path].append(original_path)

        return cls(
            source_to_inspection_map={
                key: _unique(value) for key, value in source_to_inspection.items()
            },
            inspection_to_source_map={
                key: _unique(value) for key, value in inspection_to_source.items()
            },
        )

    @property
    def enabled(self) -> bool:
        """Return whether this map contains non-empty SO correspondence."""
        return bool(self.source_to_inspection_map or self.inspection_to_source_map)

    def translate_source_to_inspection(self, path: str) -> PathTranslation:
        """Translate a source-space path into inspection-space paths."""
        inspection_paths = self._translate_with_prefix(
            path,
            self.source_to_inspection_map,
        )
        if not inspection_paths:
            inspection_paths = [path]
        return PathTranslation(
            input_path=path,
            source_paths=[path],
            inspection_paths=inspection_paths,
            ambiguous=len(inspection_paths) > 1,
        )

    def translate_inspection_to_source(self, path: str) -> PathTranslation:
        """Translate an inspection-space path into source-space paths."""
        source_paths = self._translate_with_prefix(
            path,
            self.inspection_to_source_map,
        )
        if not source_paths:
            source_paths = [path]
        return PathTranslation(
            input_path=path,
            source_paths=source_paths,
            inspection_paths=[path],
            ambiguous=len(source_paths) > 1,
        )

    @staticmethod
    def _translate_with_prefix(path: str, mapping: dict[str, list[str]]) -> list[str]:
        if path in mapping:
            return mapping[path]
        for mapped_path in sorted(
            mapping, key=lambda item: item.count("/"), reverse=True
        ):
            if path == mapped_path or path.startswith(f"{mapped_path}/"):
                suffix = path[len(mapped_path) :]
                # A dedup/split target can itself carry a nested suffix
                # relative to its own key (e.g. "mesh_I2" -> "mesh_I2/
                # Geometry"). If the query already ends with that same
                # suffix (it was already resolved once, or is being
                # translated in the wrong direction by a caller), appending
                # it again would double it into ".../Geometry/Geometry" and
                # 404 downstream. Only append when the target doesn't
                # already carry it.
                return _unique(
                    [
                        target
                        if suffix and target.endswith(suffix)
                        else (f"{target}{suffix}")
                        for target in mapping[mapped_path]
                    ]
                )
        return []

    def summary(self) -> dict[str, int]:
        """Return compact map stats for API responses."""
        ambiguous_inspection = sum(
            1 for values in self.inspection_to_source_map.values() if len(values) > 1
        )
        ambiguous_source = sum(
            1 for values in self.source_to_inspection_map.values() if len(values) > 1
        )
        return {
            "source_paths": len(self.source_to_inspection_map),
            "inspection_paths": len(self.inspection_to_source_map),
            "ambiguous_source_paths": ambiguous_source,
            "ambiguous_inspection_paths": ambiguous_inspection,
        }


def _coerce_path_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _geomsubset_paths_by_parent(original_usd_path: Path) -> dict[str, list[str]]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(original_usd_path))
    if stage is None:
        return {}
    result: dict[str, list[str]] = defaultdict(list)
    for prim in stage.TraverseAll():
        if prim.IsA(UsdGeom.Subset):
            parent = prim.GetParent()
            if parent and parent.IsValid():
                result[str(parent.GetPath())].append(str(prim.GetPath()))
    return dict(result)
