# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene manifest data model for large-scene multi-asset pipeline.

Defines SubAsset, InstanceGroup, and SceneManifest dataclasses with
JSON serialization and asset filtering logic.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SubAsset:
    """A detected sub-asset (object) within a large USD scene."""

    id: str
    name: str
    prim_path: str
    parent_group: str | None = None
    source_classification: str | None = None
    mesh_count: int = 0
    vertex_count: int = 0
    instance_group: str | None = None
    split_context: dict[str, Any] | None = None  # parent/sibling context from split

    # Processing state (updated by each step)
    extracted_usd: str | None = None
    config_path: str | None = None
    # Value-free paths for credentials omitted from the generated config.
    # Runtime values must be rehydrated from the current source scene config.
    config_credential_paths: list[str] = field(default_factory=list)
    working_dir: str | None = None
    predictions_path: str | None = None
    material_layer_path: str | None = None
    status: str = "pending"  # pending | extracted | completed | failed | skipped


@dataclass
class InstanceGroup:
    """A group of instanced sub-assets sharing the same source geometry."""

    group_name: str
    source_file: str | list[str] | None = None
    instance_count: int = 0
    member_paths: list[str] = field(default_factory=list)
    representative_id: str | None = None


@dataclass
class PayloadGroup:
    """A unique payload file in the scene's payload dependency DAG.

    Each PayloadGroup represents a single USD payload file. Payloads form
    a DAG: leaf payloads have no children, parent payloads reference other
    payload files. The pipeline processes payloads bottom-up (leaves first),
    producing an ``output.usd`` per payload that sublayers the original —
    a drop-in replacement with materials applied. Parent payloads receive
    modified copies that point to updated child versions before processing.
    """

    id: str
    group_name: str
    payload_file: str  # Absolute path to the payload USD
    instance_count: int = 0
    instance_paths: list[str] = field(
        default_factory=list
    )  # Scene paths of instance prims

    # DAG structure (populated by analyze step)
    depth: int = 0  # 0 = leaf, higher = further from leaves
    child_payload_files: list[str] = field(default_factory=list)
    parent_payload_files: list[str] = field(default_factory=list)

    # Representative file for large payloads with internal instancing.
    # Contains only the non-instance prototype source prims (much smaller).
    # Used for SO/render/predict; apply sublayers the original payload_file.
    representative_path: str | None = None

    # Processing state (updated by each step)
    config_path: str | None = None
    # Value-free paths for credentials omitted from the generated config.
    config_credential_paths: list[str] = field(default_factory=list)
    working_dir: str | None = None
    predictions_path: str | None = None
    material_layer_path: str | None = None
    modified_input_path: str | None = None  # Rewritten copy for parents
    output_usd_path: str | None = None  # The "new version" (apply step output)
    status: str = "pending"  # pending | completed | failed | skipped


@dataclass
class SceneManifest:
    """Persistent manifest tracking scene analysis and processing state."""

    schema_version: str = "1.0.0"
    scene_usd_path: str = ""
    generated_at: str = ""
    analysis: dict[str, Any] = field(default_factory=dict)
    sub_assets: list[SubAsset] = field(default_factory=list)
    instance_groups: list[InstanceGroup] = field(default_factory=list)
    payload_groups: list[PayloadGroup] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Save manifest to JSON file with atomic replacement."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                encoding="utf-8",
                delete=False,
            ) as f:
                tmp_path = Path(f.name)
                json.dump(data, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(path)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        logger.info(f"Manifest saved to {path}")

    @classmethod
    def load(cls, path: Path) -> SceneManifest:
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)

        sub_assets = [SubAsset(**sa) for sa in data.pop("sub_assets", [])]
        instance_groups = [
            InstanceGroup(**ig) for ig in data.pop("instance_groups", [])
        ]
        payload_groups = [PayloadGroup(**pg) for pg in data.pop("payload_groups", [])]
        return cls(
            sub_assets=sub_assets,
            instance_groups=instance_groups,
            payload_groups=payload_groups,
            **data,
        )

    def get_processable_assets(
        self, names_filter: list[str] | None = None
    ) -> list[SubAsset]:
        """Return assets to process, applying name/path filter and instance dedup.

        Args:
            names_filter: Optional list of asset names or prim paths to include.
                Names are matched case-insensitively. Paths starting with '/'
                are matched as prim path prefixes.

        Returns:
            List of SubAsset objects that should be processed.
        """
        # Build set of representative IDs for instance groups
        representative_ids: set[str] = set()
        instance_member_ids: set[str] = set()
        for ig in self.instance_groups:
            if ig.representative_id:
                representative_ids.add(ig.representative_id)
                # Map member paths to sub-asset IDs
                for sa in self.sub_assets:
                    if (
                        sa.prim_path in ig.member_paths
                        and sa.id != ig.representative_id
                    ):
                        instance_member_ids.add(sa.id)

        result: list[SubAsset] = []
        for sa in self.sub_assets:
            # Skip explicitly skipped assets
            if sa.status == "skipped":
                continue

            # Skip non-representative instance members (but keep assets
            # that are representatives for other instance groups — they
            # must be processed so their predictions can propagate).
            if sa.id in instance_member_ids and sa.id not in representative_ids:
                continue

            # Apply name/path filter
            if names_filter:
                matched = False
                for f in names_filter:
                    if f.startswith("/"):
                        # Path filter: match as prefix
                        if sa.prim_path == f or sa.prim_path.startswith(f + "/"):
                            matched = True
                            break
                    else:
                        # Name filter: case-insensitive match
                        if sa.name.lower() == f.lower():
                            matched = True
                            break
                if not matched:
                    continue

            result.append(sa)

        return result

    def get_processable_payloads(self) -> list[PayloadGroup]:
        """Return payload groups that should be processed.

        Returns all payload groups with status != "skipped".
        """
        return [pg for pg in self.payload_groups if pg.status != "skipped"]

    def get_payloads_by_depth(self) -> dict[int, list[PayloadGroup]]:
        """Group processable payloads by depth level.

        Returns dict mapping depth -> list of PayloadGroups at that depth.
        Depth 0 = leaves (no children), higher = further from leaves.
        """
        by_depth: dict[int, list[PayloadGroup]] = {}
        for pg in self.get_processable_payloads():
            by_depth.setdefault(pg.depth, []).append(pg)
        return by_depth

    def get_payload_by_file(self, payload_file: str) -> PayloadGroup | None:
        """Find a payload group by its payload file path."""
        resolved = str(Path(payload_file).resolve())
        for pg in self.payload_groups:
            if str(Path(pg.payload_file).resolve()) == resolved:
                return pg
        return None

    def get_asset_by_id(self, asset_id: str) -> SubAsset | None:
        """Find a sub-asset by its ID."""
        for sa in self.sub_assets:
            if sa.id == asset_id:
                return sa
        return None

    def get_instance_group(self, group_name: str) -> InstanceGroup | None:
        """Find an instance group by name."""
        for ig in self.instance_groups:
            if ig.group_name == group_name:
                return ig
        return None

    @staticmethod
    def timestamp() -> str:
        """Generate an ISO 8601 timestamp."""
        return datetime.now(UTC).isoformat()
