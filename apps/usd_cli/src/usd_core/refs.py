# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressed reference system (PR-1) — deterministic by SdfPath.

Unlike a browser connector, whose refs are fully
ephemeral and reset to 1 on every snapshot, usd-cli assigns refs **deterministically by SdfPath and
keeps them stable within a session**. A prim that still exists at its path gets the
*same* `@n` across snapshots; only new paths mint new ids. The table resets only on
`open`/reload (PR-1.4).

Why: SdfPath is semantically stable (the DOM had no stable id, so browse couldn't do
this). Stable refs make `snapshot -D` diffs clean — only real changes show — and let
undo restore a prim's ref for free: undo brings the prim back to its path, and the path
reclaims its old ref with no ref-table surgery.

v1 fork resolution: rename/reparent *changes* a prim's path, so the old ref goes stale
and re-snapshot reassigns (rather than the ref following the prim).
"""

from __future__ import annotations

from dataclasses import dataclass

#: ref prefix -> human label (PR-1.2)
NAMESPACES = {
    "n": "node/prim",
    "m": "material",
    "l": "light",
    "c": "camera",
    "a": "animation",
    "s": "space/region",
}

#: Namespaces whose refs do NOT name a prim on the stage.
#:
#: `@s` regions are *derived results* — free space found by `space free/support/inside`
#: — so they have no SdfPath, and the liveness rule that makes a `@n` ref stale (its
#: prim was deleted) does not apply. The Session stale-checks them against the mutation
#: epoch instead: any edit to the stage invalidates a previously reported free region,
#: because the geometry it was measured against has changed. Everything else — minting,
#: numbering, `resolve`, reset-on-open — is shared with prim refs on purpose, so `@s1`
#: is an addressable handle anywhere a positional target is accepted.
DERIVED_NAMESPACES = frozenset({"s"})


class StaleRefError(Exception):
    """Raised when a ref no longer denotes what it did (PR-1.5).

    Recovery differs by namespace, and the message has to say which: a stale `@n`
    is refreshed by `snapshot`, but `snapshot` will never mint a `@s` — those come
    only from re-running the `space` query that produced them. Pointing an agent at
    the wrong recovery command costs it a whole loop.
    """

    def __init__(self, ref: str, last_path: str = "", reason: str = "deleted"):
        self.ref = ref
        self.last_path = last_path
        ns = ref[1:2] if ref.startswith("@") else ""
        if ns in DERIVED_NAMESPACES:
            detail = f"{reason}. " if reason and reason != "deleted" else ""
            recovery = "re-run 'space free' / 'space support' to mint fresh region refs."
        else:
            detail = f"prim '{last_path}' was {reason}. " if last_path else ""
            recovery = "run 'snapshot' for fresh refs."
        super().__init__(f"ref {ref} is stale — {detail}{recovery}")


@dataclass
class RefEntry:
    path: str  # full SdfPath
    type: str  # prim type, e.g. "Mesh"
    name: str  # display name


class RefTable:
    """Session-scoped, deterministic-by-path. Stable across snapshots; reset on reload."""

    def __init__(self) -> None:
        self._by_ref: dict[str, RefEntry] = {}
        self._path_to_ref: dict[str, str] = {}
        # Derived results are keyed within their own namespace.  Their display path is
        # deliberately USD-shaped for `resolve`, but a stage may legally contain the
        # same path; sharing the prim path index would then turn an `@s` into `@n`.
        self._derived_path_to_ref: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {k: 0 for k in NAMESPACES}

    def assign(self, ns: str, path: str, type_: str, name: str) -> str:
        """Return the stable ref for `path`, minting a new id only if unseen this session.

        Re-assigning a known path reuses its ref and refreshes the metadata — this is what
        keeps refs stable across snapshots and lets undo reclaim a ref by restoring a path.
        """
        key = (ns, path)
        existing = (self._derived_path_to_ref.get(key)
                    if ns in DERIVED_NAMESPACES else self._path_to_ref.get(path))
        if existing is not None:
            self._by_ref[existing] = RefEntry(path=path, type=type_, name=name)
            return existing
        if ns not in NAMESPACES:
            raise ValueError(f"unknown ref namespace '@{ns}' (expected one of {list(NAMESPACES)})")
        self._counters[ns] += 1
        ref = f"@{ns}{self._counters[ns]}"
        self._by_ref[ref] = RefEntry(path=path, type=type_, name=name)
        if ns in DERIVED_NAMESPACES:
            self._derived_path_to_ref[key] = ref
        else:
            self._path_to_ref[path] = ref
        return ref

    def resolve(self, ref: str) -> RefEntry:
        """Look up a ref's last-known entry. Liveness (prim.IsValid) is checked by the
        Session against the stage, which raises StaleRefError with the recovery hint."""
        entry = self._by_ref.get(ref)
        if entry is None:
            raise StaleRefError(ref)
        return entry

    def ref_for_path(self, path: str) -> str | None:
        return self._path_to_ref.get(path)

    def reset(self) -> None:
        """Drop all refs — only on open/reload (PR-1.4)."""
        self.__init__()

    def reset_namespace(self, ns: str) -> None:
        """Drop every ref in one namespace and restart its counter at 1.

        Only meaningful for derived namespaces. A `space` query is a *result set*,
        not a set of scene entities: re-running it must renumber from `@s1` in the
        new ranking rather than append `@s11…@s20` to the previous run's refs, or
        the numbering drifts upward every query and `@s1` stops meaning "the best
        candidate". Prim refs must never be reset this way — their stability across
        snapshots is the whole point of the ref system (PR-1.4).
        """
        if ns not in DERIVED_NAMESPACES:
            raise ValueError(
                f"refusing to reset '@{ns}': only derived namespaces "
                f"({sorted(DERIVED_NAMESPACES)}) may be renumbered; prim refs are "
                "stable within a session by design")
        stale = [r for r in self._by_ref if r.startswith(f"@{ns}")]
        for ref in stale:
            entry = self._by_ref.pop(ref)
            self._derived_path_to_ref.pop((ns, entry.path), None)
        self._counters[ns] = 0

    def __len__(self) -> int:
        return len(self._by_ref)
