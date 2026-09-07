# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
USD Model - A queryable data model for USD stage hierarchies.

This module provides classes to load, represent, and query USD stage structures,
including prim relationships, collections, transforms, and metadata.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pxr import Sdf, Usd, UsdGeom  # type: ignore

from world_understanding.utils.credentials import redact_sensitive_log_text

logger = logging.getLogger(__name__)


@dataclass
class VariantSelection:
    """Represents a variant set and its selection."""

    set_name: str
    selection: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary representation."""
        return {"set_name": self.set_name, "selection": self.selection}


@dataclass
class CollectionInfo:
    """Represents a collection defined on a prim."""

    name: str
    prim_path: str  # Path of the prim that defines this collection
    includes: list[str] = field(default_factory=list)  # Included prim paths
    excludes: list[str] = field(default_factory=list)  # Excluded prim paths

    def contains_prim(self, prim_path: str) -> bool:
        """Check if a prim path is included in this collection."""
        # Check if the prim or any of its parents are in includes
        for include_path in self.includes:
            # Ensure we match complete path segments
            if prim_path == include_path or prim_path.startswith(include_path + "/"):
                # Check if it's not excluded
                for exclude_path in self.excludes:
                    if prim_path == exclude_path or prim_path.startswith(
                        exclude_path + "/"
                    ):
                        return False
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "prim_path": self.prim_path,
            "includes": self.includes,
            "excludes": self.excludes,
        }


@dataclass
class USDPrimNode:
    """Represents a single USD prim with its metadata and relationships."""

    path: str
    name: str
    type_name: str | None = None
    is_active: bool = True
    is_instance: bool = False
    is_in_prototype: bool = False
    is_xform: bool = False
    parent_path: str | None = None
    children_paths: list[str] = field(default_factory=list)

    # Metadata
    variant_selections: list[VariantSelection] = field(default_factory=list)
    api_schemas: list[str] = field(default_factory=list)
    custom_tokens: dict[str, str] = field(default_factory=dict)

    # Collections defined ON this prim
    defined_collections: list[CollectionInfo] = field(default_factory=list)

    # Additional USD data
    attributes: dict[str, Any] = field(default_factory=dict)
    properties: dict[str, Any] = field(default_factory=dict)

    def get_depth(self) -> int:
        """Get the depth of this prim in the hierarchy."""
        return self.path.count("/")

    def is_descendant_of(self, ancestor_path: str) -> bool:
        """Check if this prim is a descendant of the given path."""
        return self.path.startswith(ancestor_path) and self.path != ancestor_path

    def is_ancestor_of(self, descendant_path: str) -> bool:
        """Check if this prim is an ancestor of the given path."""
        return descendant_path.startswith(self.path) and descendant_path != self.path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "name": self.name,
            "type_name": self.type_name,
            "is_active": self.is_active,
            "is_instance": self.is_instance,
            "is_in_prototype": self.is_in_prototype,
            "is_xform": self.is_xform,
            "parent_path": self.parent_path,
            "children_paths": self.children_paths,
            "variant_selections": [v.to_dict() for v in self.variant_selections],
            "api_schemas": self.api_schemas,
            "custom_tokens": self.custom_tokens,
            "defined_collections": [c.to_dict() for c in self.defined_collections],
            "attributes": self.attributes,
            "properties": self.properties,
        }


class USDModel:
    """
    A queryable model of a USD stage hierarchy.

    This class loads a USD stage and builds an internal representation that
    allows for efficient querying of prim relationships, collections, and metadata.
    """

    def __init__(self, usd_file: str | Path, load_stage: bool = True):
        """
        Initialize the USD model.

        Args:
            usd_file: Path to the USD file
            load_stage: Whether to immediately load the stage (default True)
        """
        self.usd_file = Path(usd_file) if isinstance(usd_file, str) else usd_file
        self.stage: Usd.Stage | None = None  # type: ignore
        self.prims: dict[str, USDPrimNode] = {}
        self.collections: list[CollectionInfo] = []

        # Indexes for efficient querying
        self._prims_by_type: dict[str, set[str]] = {}
        self._prims_by_name: dict[str, set[str]] = {}
        self._collection_membership: dict[
            str, set[str]
        ] = {}  # prim_path -> collection names

        # Stage metadata
        self.root_layer: str | None = None
        self.default_prim_path: str | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.fps: float | None = None
        self.up_axis: str | None = None
        self.meters_per_unit: float | None = None

        if load_stage:
            self.load()

    def load(self) -> None:
        """Load the USD stage and build the internal model."""
        if not self.usd_file.exists():
            raise FileNotFoundError(f"USD file not found: {self.usd_file}")

        try:
            self.stage = Usd.Stage.Open(str(self.usd_file))  # type: ignore
            if not self.stage:
                raise RuntimeError(f"Failed to open USD file: {self.usd_file}")

            self._load_stage_metadata()
            self._build_prim_hierarchy()
            self._build_indexes()
            self._compute_collection_membership()

            logger.info(
                f"Loaded USD model with {len(self.prims)} prims and "
                f"{len(self.collections)} collections"
            )

        except Exception as e:
            logger.error(f"Error loading USD file: {e}")
            raise

    def _load_stage_metadata(self) -> None:
        """Load stage-level metadata."""
        if not self.stage:
            return

        self.root_layer = self.stage.GetRootLayer().identifier

        default_prim = self.stage.GetDefaultPrim()
        if default_prim:
            self.default_prim_path = str(default_prim.GetPath())

        self.start_time = self.stage.GetStartTimeCode()
        self.end_time = self.stage.GetEndTimeCode()
        self.fps = self.stage.GetFramesPerSecond()
        self.up_axis = str(UsdGeom.GetStageUpAxis(self.stage))  # type: ignore
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)  # type: ignore

    def _build_prim_hierarchy(self) -> None:
        """Build the internal prim hierarchy from the stage."""
        if not self.stage:
            return

        # Process all prims
        for prim in self.stage.Traverse():
            prim_node = self._create_prim_node(prim)
            self.prims[prim_node.path] = prim_node

            # Track collections
            for collection in prim_node.defined_collections:
                self.collections.append(collection)

        # Build parent-child relationships
        for path, prim_node in self.prims.items():
            parent_path = str(Sdf.Path(path).GetParentPath())  # type: ignore
            if parent_path in self.prims:
                prim_node.parent_path = parent_path
                self.prims[parent_path].children_paths.append(path)

    def _create_prim_node(self, prim) -> USDPrimNode:
        """Create a USDPrimNode from a USD prim."""
        path = str(prim.GetPath())
        node = USDPrimNode(
            path=path,
            name=prim.GetName(),
            type_name=str(prim.GetTypeName()) if prim.GetTypeName() else None,
            is_active=prim.IsActive(),
            is_instance=prim.IsInstance(),
            is_in_prototype=prim.IsInPrototype(),
            is_xform=prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Xformable),  # type: ignore
        )

        # Get variant selections
        if prim.HasVariantSets():
            variant_sets = prim.GetVariantSets()
            for set_name in variant_sets.GetNames():
                variant_set = variant_sets.GetVariantSet(set_name)
                selection = variant_set.GetVariantSelection()
                if selection:
                    node.variant_selections.append(
                        VariantSelection(set_name, selection)
                    )

        # Get API schemas
        node.api_schemas = [str(s) for s in prim.GetAppliedSchemas()]

        # Get custom tokens
        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            if ":" in attr_name and attr.GetTypeName() == Sdf.ValueTypeNames.Token:  # type: ignore
                value = attr.Get()
                if value is not None:
                    node.custom_tokens[attr_name] = str(value)

        # Get collections defined on this prim
        collection_apis = Usd.CollectionAPI.GetAll(prim)  # type: ignore
        for collection_api in collection_apis:
            collection_info = CollectionInfo(
                name=collection_api.GetName(), prim_path=path
            )

            # Get includes
            includes_rel = collection_api.GetIncludesRel()
            if includes_rel:
                collection_info.includes = [str(t) for t in includes_rel.GetTargets()]

            # Get excludes
            excludes_rel = collection_api.GetExcludesRel()
            if excludes_rel:
                collection_info.excludes = [str(t) for t in excludes_rel.GetTargets()]

            node.defined_collections.append(collection_info)

        return node

    def _build_indexes(self) -> None:
        """Build indexes for efficient querying."""
        self._prims_by_type.clear()
        self._prims_by_name.clear()

        for path, prim_node in self.prims.items():
            # Index by type
            if prim_node.type_name:
                if prim_node.type_name not in self._prims_by_type:
                    self._prims_by_type[prim_node.type_name] = set()
                self._prims_by_type[prim_node.type_name].add(path)

            # Index by name
            if prim_node.name not in self._prims_by_name:
                self._prims_by_name[prim_node.name] = set()
            self._prims_by_name[prim_node.name].add(path)

    def _compute_collection_membership(self) -> None:
        """Compute which prims belong to which collections."""
        self._collection_membership.clear()

        for collection in self.collections:
            collection_id = f"{collection.prim_path}:{collection.name}"

            for prim_path in self.prims.keys():
                if collection.contains_prim(prim_path):
                    if prim_path not in self._collection_membership:
                        self._collection_membership[prim_path] = set()
                    self._collection_membership[prim_path].add(collection_id)

    # Query methods

    def get_prim(self, path: str) -> USDPrimNode | None:
        """Get a prim node by its path."""
        return self.prims.get(path)

    def get_prims_by_type(self, type_name: str) -> list[USDPrimNode]:
        """Get all prims of a specific type."""
        paths = self._prims_by_type.get(type_name, set())
        return [self.prims[p] for p in paths]

    def get_prims_by_name(self, name: str) -> list[USDPrimNode]:
        """Get all prims with a specific name."""
        paths = self._prims_by_name.get(name, set())
        return [self.prims[p] for p in paths]

    def get_all_xforms(self) -> list[USDPrimNode]:
        """Get all Xform prims in the stage."""
        return [p for p in self.prims.values() if p.is_xform]

    def get_all_meshes(self) -> list[USDPrimNode]:
        """Get all Mesh prims in the stage."""
        return self.get_prims_by_type("Mesh")

    def get_parent(self, prim_path: str) -> USDPrimNode | None:
        """Get the parent of a prim."""
        prim = self.get_prim(prim_path)
        if prim and prim.parent_path:
            return self.get_prim(prim.parent_path)
        return None

    def get_children(self, prim_path: str) -> list[USDPrimNode]:
        """Get the children of a prim."""
        prim = self.get_prim(prim_path)
        if prim:
            return [self.prims[p] for p in prim.children_paths if p in self.prims]
        return []

    def get_ancestors(
        self, prim_path: str, include_self: bool = False
    ) -> list[USDPrimNode]:
        """Get all ancestors of a prim (from parent to root)."""
        ancestors = []
        current_path = prim_path if include_self else None

        if not include_self:
            prim = self.get_prim(prim_path)
            if prim:
                current_path = prim.parent_path

        while current_path and current_path in self.prims:
            ancestors.append(self.prims[current_path])
            parent = self.get_parent(current_path)
            current_path = parent.path if parent else None

        return ancestors

    def get_descendants(
        self, prim_path: str, include_self: bool = False
    ) -> list[USDPrimNode]:
        """Get all descendants of a prim."""
        descendants = []

        if include_self and prim_path in self.prims:
            descendants.append(self.prims[prim_path])

        # Use path prefix matching for efficient descendant finding
        for _path, prim in self.prims.items():
            if prim.is_descendant_of(prim_path):
                descendants.append(prim)

        return descendants

    def get_collections_containing_prim(self, prim_path: str) -> list[CollectionInfo]:
        """Get all collections that contain the specified prim."""
        collection_ids = self._collection_membership.get(prim_path, set())
        collections = []

        for collection_id in collection_ids:
            # Parse collection_id (format: "prim_path:collection_name")
            for collection in self.collections:
                if f"{collection.prim_path}:{collection.name}" == collection_id:
                    collections.append(collection)
                    break

        return collections

    def get_xform_owning_collection(
        self, collection: CollectionInfo
    ) -> USDPrimNode | None:
        """
        Get the Xform that owns (defines) a collection.

        Args:
            collection: The collection to find the owner for

        Returns:
            The Xform prim that owns the collection, or None if not owned by an Xform
        """
        prim = self.get_prim(collection.prim_path)
        if prim and prim.is_xform:
            return prim

        # Check if any ancestor is an Xform
        ancestors = self.get_ancestors(collection.prim_path)
        for ancestor in ancestors:
            if ancestor.is_xform:
                return ancestor

        return None

    def get_collections_on_prim(self, prim_path: str) -> list[CollectionInfo]:
        """Get all collections defined ON a specific prim."""
        prim = self.get_prim(prim_path)
        if prim:
            return prim.defined_collections.copy()
        return []

    def find_prim_in_collection_of_xform(
        self, prim_path: str
    ) -> list[tuple[CollectionInfo, USDPrimNode]]:
        """
        Find which collections contain a prim and which Xforms own those collections.

        Args:
            prim_path: Path of the prim to check

        Returns:
            List of tuples (collection, xform) where the prim is in the collection
            and the xform owns that collection
        """
        results = []
        collections = self.get_collections_containing_prim(prim_path)

        for collection in collections:
            xform = self.get_xform_owning_collection(collection)
            if xform:
                results.append((collection, xform))

        return results

    def get_path_to_root(self, prim_path: str) -> list[str]:
        """Get the path from a prim to the root as a list of prim paths."""
        path = []
        current = self.get_prim(prim_path)

        while current:
            path.append(current.path)
            current = self.get_parent(current.path)

        return path

    def get_subtree_stats(self, root_path: str) -> dict[str, Any]:
        """Get statistics about a subtree rooted at the given path."""
        descendants = self.get_descendants(root_path, include_self=True)

        type_counts: dict[str, int] = {}
        for prim in descendants:
            type_name = prim.type_name or "<no type>"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        return {
            "total_prims": len(descendants),
            "type_counts": type_counts,
            "max_depth": max((p.get_depth() for p in descendants), default=0),
            "num_xforms": sum(1 for p in descendants if p.is_xform),
            "num_instances": sum(1 for p in descendants if p.is_instance),
            "num_prototype_prims": sum(1 for p in descendants if p.is_in_prototype),
            "num_inactive": sum(1 for p in descendants if not p.is_active),
        }

    def print_summary(self) -> None:
        """Print a summary of the USD model."""
        print(f"USD Model Summary for: {self.usd_file}")
        print(f"  Root Layer: {self.root_layer}")
        print(f"  Default Prim: {self.default_prim_path}")
        print(f"  Total Prims: {len(self.prims)}")
        print(f"  Total Collections: {len(self.collections)}")

        # Type distribution
        type_counts: dict[str, int] = {}
        for prim in self.prims.values():
            type_name = prim.type_name or "<no type>"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        print("\n  Top Prim Types:")
        for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    {type_name}: {count}")

        print(f"\n  Xforms: {len(self.get_all_xforms())}")
        print(f"  Meshes: {len(self.get_all_meshes())}")

    def _get_stage_info_str(self) -> str:
        """Get basic information about the USD stage as a string."""
        lines = []
        lines.append("Stage Information:")
        lines.append(f"  Root Layer: {self.root_layer}")

        if self.default_prim_path:
            lines.append(f"  Default Prim: {self.default_prim_path}")

        lines.append(
            f"  Time Range: {self.start_time} - {self.end_time} @ {self.fps} fps"
        )
        lines.append(f"  Up Axis: {self.up_axis}")
        lines.append(f"  Meters Per Unit: {self.meters_per_unit}")
        lines.append("")
        return "\n".join(lines)

    def print_stage_info(self) -> None:
        """Print basic information about the USD stage."""
        print(self._get_stage_info_str(), end="")

    def print_tree_to_str(
        self,
        start_path: str | None = None,
        show_types: bool = True,
        show_variants: bool = False,
        show_api_schemas: bool = False,
        show_collections: bool = False,
        show_custom_tokens: bool = False,
        active_only: bool = False,
        max_depth: int | None = None,
        show_info: bool = True,
        show_stats: bool = False,
    ) -> str:
        """
        Generate the USD hierarchy as a tree string.

        Args:
            start_path: Start traversal from specific prim path (None for root)
            show_types: Whether to show prim types
            show_variants: Whether to show variant sets and selections
            show_api_schemas: Whether to show applied API schemas
            show_collections: Whether to show collections defined on prims
            show_custom_tokens: Whether to show custom token attributes
            active_only: Whether to show only active prims
            max_depth: Maximum depth to traverse (None for unlimited)
            show_info: Whether to show stage information header
            show_stats: Whether to show statistics after the tree

        Returns:
            String representation of the USD hierarchy tree
        """
        lines = []

        if show_info:
            lines.append(self._get_stage_info_str())

        lines.append("Scene Hierarchy:")

        # Determine starting point
        if start_path:
            if start_path not in self.prims:
                logger.error(f"Invalid prim path: {start_path}")
                return "\n".join(lines) + f"\nError: Invalid prim path: {start_path}"
            root_prim = self.prims[start_path]
            tree_lines = self._get_prim_tree_lines(
                root_prim,
                "",
                True,
                show_types,
                show_variants,
                show_api_schemas,
                show_collections,
                show_custom_tokens,
                active_only,
                max_depth,
                0,
            )
            lines.extend(tree_lines)
        else:
            # Find root-level prims (those without parents or with "/" parent)
            root_prims = []
            for prim in self.prims.values():
                if not prim.parent_path or prim.parent_path == "/":
                    root_prims.append(prim)

            # Sort by path for consistent ordering
            root_prims.sort(key=lambda p: p.path)

            for i, prim in enumerate(root_prims):
                is_last = i == len(root_prims) - 1
                tree_lines = self._get_prim_tree_lines(
                    prim,
                    "",
                    is_last,
                    show_types,
                    show_variants,
                    show_api_schemas,
                    show_collections,
                    show_custom_tokens,
                    active_only,
                    max_depth,
                    0,
                )
                lines.extend(tree_lines)

        if show_stats:
            lines.append(self._get_statistics_str())

        return "\n".join(lines)

    def print_tree(
        self,
        start_path: str | None = None,
        show_types: bool = True,
        show_variants: bool = False,
        show_api_schemas: bool = False,
        show_collections: bool = False,
        show_custom_tokens: bool = False,
        active_only: bool = False,
        max_depth: int | None = None,
        show_info: bool = True,
        show_stats: bool = False,
    ) -> None:
        """
        Print the USD hierarchy as a tree.

        Args:
            start_path: Start traversal from specific prim path (None for root)
            show_types: Whether to show prim types
            show_variants: Whether to show variant sets and selections
            show_api_schemas: Whether to show applied API schemas
            show_collections: Whether to show collections defined on prims
            show_custom_tokens: Whether to show custom token attributes
            active_only: Whether to show only active prims
            max_depth: Maximum depth to traverse (None for unlimited)
            show_info: Whether to show stage information header
            show_stats: Whether to show statistics after the tree
        """
        tree_str = self.print_tree_to_str(
            start_path=start_path,
            show_types=show_types,
            show_variants=show_variants,
            show_api_schemas=show_api_schemas,
            show_collections=show_collections,
            show_custom_tokens=show_custom_tokens,
            active_only=active_only,
            max_depth=max_depth,
            show_info=show_info,
            show_stats=show_stats,
        )
        print(tree_str)

    def _get_prim_tree_lines(
        self,
        prim_node: USDPrimNode,
        prefix: str,
        is_last: bool,
        show_types: bool,
        show_variants: bool,
        show_api_schemas: bool,
        show_collections: bool,
        show_custom_tokens: bool,
        active_only: bool,
        max_depth: int | None,
        current_depth: int,
    ) -> list[str]:
        """
        Recursively generate prim and its children as tree lines.

        Internal method used by print_tree_to_str().

        Returns:
            List of strings representing tree lines
        """
        lines = []

        if max_depth is not None and current_depth > max_depth:
            return lines

        # Skip inactive prims if active_only is True
        if active_only and not prim_node.is_active:
            return lines

        # Prepare the tree characters
        connector = "└── " if is_last else "├── "

        # Build the output string
        output = prefix + connector + prim_node.name

        # Add prim type if requested
        if show_types and prim_node.type_name:
            output += f" [{prim_node.type_name}]"

        # Add variant sets if requested
        if show_variants and prim_node.variant_selections:
            variants = [
                f"{v.set_name}={v.selection}" for v in prim_node.variant_selections
            ]
            output += f" {{{', '.join(variants)}}}"

        # Add API schemas if requested
        if show_api_schemas and prim_node.api_schemas:
            output += f" <{', '.join(prim_node.api_schemas)}>"

        # Add collections if requested
        if show_collections and prim_node.defined_collections:
            collection_strs = []
            for collection in prim_node.defined_collections:
                if collection.includes:
                    target_list = ", ".join(collection.includes)
                    collection_info = f"{collection.name}:[{target_list}]"
                else:
                    collection_info = f"{collection.name}:[]"
                collection_strs.append(collection_info)
            output += f" {{{'; '.join(collection_strs)}}}"

        # Add custom tokens if requested
        if show_custom_tokens and prim_node.custom_tokens:
            token_strs = [f"{k}={v}" for k, v in prim_node.custom_tokens.items()]
            output += f" |{', '.join(token_strs)}|"

        # Add active state indicator if not showing only active prims
        if not active_only and not prim_node.is_active:
            output += " (inactive)"

        lines.append(output)

        # Prepare the prefix for children
        extension = "    " if is_last else "│   "
        child_prefix = prefix + extension

        # Get children
        children = self.get_children(prim_node.path)
        if children:
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                child_lines = self._get_prim_tree_lines(
                    child,
                    child_prefix,
                    is_last_child,
                    show_types,
                    show_variants,
                    show_api_schemas,
                    show_collections,
                    show_custom_tokens,
                    active_only,
                    max_depth,
                    current_depth + 1,
                )
                lines.extend(child_lines)

        return lines

    def _print_prim_tree(
        self,
        prim_node: USDPrimNode,
        prefix: str,
        is_last: bool,
        show_types: bool,
        show_variants: bool,
        show_api_schemas: bool,
        show_collections: bool,
        show_custom_tokens: bool,
        active_only: bool,
        max_depth: int | None,
        current_depth: int,
    ) -> None:
        """
        Recursively print a prim and its children as a tree.

        Internal method used by print_tree().
        """
        lines = self._get_prim_tree_lines(
            prim_node,
            prefix,
            is_last,
            show_types,
            show_variants,
            show_api_schemas,
            show_collections,
            show_custom_tokens,
            active_only,
            max_depth,
            current_depth,
        )
        for line in lines:
            print(line)

    def _get_statistics_str(self) -> str:
        """Get statistics about the USD model as a string."""
        lines = []
        lines.append("\nStatistics:")
        lines.append(f"  Total Prims: {len(self.prims)}")

        # Count by type
        type_counts: dict[str, int] = {}
        for prim in self.prims.values():
            type_name = prim.type_name or "<no type>"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1

        if type_counts:
            lines.append("\n  Prim Types:")
            for prim_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {prim_type}: {count}")

        # Count meshes
        mesh_count = len(self.get_all_meshes())
        if mesh_count:
            lines.append(f"\n  Total Meshes: {mesh_count}")

        # Count instances
        instance_count = sum(1 for p in self.prims.values() if p.is_instance)
        if instance_count:
            lines.append(f"  Total Instances: {instance_count}")

        # Count prototype prims
        prototype_count = sum(1 for p in self.prims.values() if p.is_in_prototype)
        if prototype_count:
            lines.append(f"  Total Prototype Prims: {prototype_count}")

        # Count Xforms
        xform_count = len(self.get_all_xforms())
        if xform_count:
            lines.append(f"  Total Xforms: {xform_count}")

        # Count collections
        if self.collections:
            lines.append(f"  Total Collections: {len(self.collections)}")

        return "\n".join(lines)

    def _print_statistics(self) -> None:
        """Print statistics about the USD model."""
        print(self._get_statistics_str())

    def to_dict(self, include_hierarchy: bool = True) -> dict[str, Any]:
        """
        Convert the USD model to a dictionary representation.

        Args:
            include_hierarchy: Whether to include the full prim hierarchy

        Returns:
            Dictionary containing the USD model data
        """
        result: dict[str, Any] = {
            "file_path": str(self.usd_file),
            "stage_info": {
                "root_layer": self.root_layer,
                "default_prim_path": self.default_prim_path,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "fps": self.fps,
                "up_axis": self.up_axis,
                "meters_per_unit": self.meters_per_unit,
            },
            "statistics": {
                "total_prims": len(self.prims),
                "total_collections": len(self.collections),
                "total_xforms": len(self.get_all_xforms()),
                "total_meshes": len(self.get_all_meshes()),
                "total_instances": sum(1 for p in self.prims.values() if p.is_instance),
                "total_prototype_prims": sum(
                    1 for p in self.prims.values() if p.is_in_prototype
                ),
            },
        }

        # Add type distribution
        type_counts: dict[str, int] = {}
        for prim in self.prims.values():
            type_name = prim.type_name or "<no type>"
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        result["statistics"]["type_distribution"] = type_counts

        # Add collections
        result["collections"] = [c.to_dict() for c in self.collections]

        # Add hierarchy if requested
        if include_hierarchy:
            result["prims"] = {
                path: prim.to_dict() for path, prim in self.prims.items()
            }

        return result

    def to_json(self, include_hierarchy: bool = True, indent: int | None = 2) -> str:
        """
        Convert the USD model to a JSON string.

        Args:
            include_hierarchy: Whether to include the full prim hierarchy
            indent: Indentation for pretty printing (None for compact)

        Returns:
            JSON string representation of the USD model
        """
        return json.dumps(self.to_dict(include_hierarchy), indent=indent)

    def save_json(
        self,
        file_path: str | Path,
        include_hierarchy: bool = True,
        indent: int | None = 2,
    ) -> None:
        """
        Save the USD model to a JSON file.

        Args:
            file_path: Path to save the JSON file
            include_hierarchy: Whether to include the full prim hierarchy
            indent: Indentation for pretty printing (None for compact)
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(include_hierarchy, indent))
        logger.info(f"Saved USD model to JSON: {file_path}")

    @classmethod
    def from_dict(cls, data: dict[str, Any], load_stage: bool = False) -> "USDModel":
        """
        Create a USDModel from a dictionary representation.

        Note: This creates the model structure but does NOT recreate the actual USD stage.
        It's useful for examining the structure without needing the original USD file.

        Args:
            data: Dictionary containing the USD model data
            load_stage: Whether to try loading the original USD file (if path exists)

        Returns:
            USDModel instance
        """
        file_path = data.get("file_path", "unknown.usd")
        model = cls(file_path, load_stage=False)

        # Restore stage info
        stage_info = data.get("stage_info", {})
        model.root_layer = stage_info.get("root_layer")
        model.default_prim_path = stage_info.get("default_prim_path")
        model.start_time = stage_info.get("start_time")
        model.end_time = stage_info.get("end_time")
        model.fps = stage_info.get("fps")
        model.up_axis = stage_info.get("up_axis")
        model.meters_per_unit = stage_info.get("meters_per_unit")

        # Restore prims if present
        if "prims" in data:
            for path, prim_data in data["prims"].items():
                prim = USDPrimNode(
                    path=prim_data["path"],
                    name=prim_data["name"],
                    type_name=prim_data.get("type_name"),
                    is_active=prim_data.get("is_active", True),
                    is_instance=prim_data.get("is_instance", False),
                    is_in_prototype=prim_data.get("is_in_prototype", False),
                    is_xform=prim_data.get("is_xform", False),
                    parent_path=prim_data.get("parent_path"),
                    children_paths=prim_data.get("children_paths", []),
                    variant_selections=[
                        VariantSelection(v["set_name"], v["selection"])
                        for v in prim_data.get("variant_selections", [])
                    ],
                    api_schemas=prim_data.get("api_schemas", []),
                    custom_tokens=prim_data.get("custom_tokens", {}),
                    defined_collections=[
                        CollectionInfo(
                            name=c["name"],
                            prim_path=c["prim_path"],
                            includes=c.get("includes", []),
                            excludes=c.get("excludes", []),
                        )
                        for c in prim_data.get("defined_collections", [])
                    ],
                    attributes=prim_data.get("attributes", {}),
                    properties=prim_data.get("properties", {}),
                )
                model.prims[path] = prim

        # Restore collections
        if "collections" in data:
            for coll_data in data["collections"]:
                collection = CollectionInfo(
                    name=coll_data["name"],
                    prim_path=coll_data["prim_path"],
                    includes=coll_data.get("includes", []),
                    excludes=coll_data.get("excludes", []),
                )
                model.collections.append(collection)

        # Rebuild indexes and collection membership
        if model.prims:
            model._build_indexes()
            model._compute_collection_membership()

        # Optionally try to load the actual stage if file exists
        if load_stage and Path(file_path).exists():
            try:
                model.load()
            except Exception as e:
                logger.warning(
                    "Could not load USD stage from %s: %s",
                    redact_sensitive_log_text(file_path),
                    redact_sensitive_log_text(e),
                )

        return model

    @classmethod
    def load_json(cls, file_path: str | Path, load_stage: bool = False) -> "USDModel":
        """
        Load a USDModel from a JSON file.

        Args:
            file_path: Path to the JSON file
            load_stage: Whether to try loading the original USD file

        Returns:
            USDModel instance
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, load_stage)
