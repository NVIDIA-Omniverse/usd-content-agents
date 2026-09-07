# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Undo/redo op-log (PR-7.1–7.3) — the substrate a stateless connector never had.

Design decision: browser-automation connectors have *no general undo* —
they offload to the browser's Back button. usd-cli authors a persistent USD stage, so it must
own undo. The
mechanism is a per-command op-log over a USD edit layer, using a **hybrid** scheme:

- cheap reversible verbs (transform/show/hide/rename/material) record an **inverse op**;
- destructive verbs (delete/import/cut) record a **saved subtree** (a layer fragment to
  restore), since they can't be inverted from parameters alone.

This module is the bookkeeping half — generic and stage-free, so it's unit-testable. The
Session owns the other half: producing the inverse / saved subtree when recording, and
re-applying it to the stage on undo.

Granularity: one entry per command. A `chain`/`batch` opens a transaction so the whole
group collapses into a single compound undo entry (PR-12, `chain --atomic`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Op:
    """One reversible unit of work."""

    command: str
    label: str = ""
    inverse: dict | None = None  # inverse op payload (cheap reversible verbs)
    saved_subtree: str | None = None  # path to a saved layer fragment (destructive verbs)
    children: list["Op"] = field(default_factory=list)  # for compound (transaction) entries
    #: honesty marker: set (to a short reason, e.g. "remove-api") when this entry cannot
    #: be reversed. `undo` still pops it, but REPORTS a warning instead of pretending the
    #: stage was restored — a fake `set_active(True)→set_active(True)` inverse used to
    #: make undo claim success while restoring nothing.
    non_undoable: str | None = None

    @property
    def is_compound(self) -> bool:
        return bool(self.children)


class History:
    """Undo/redo stacks with transaction support and a bounded window."""

    def __init__(self, limit: int = 200) -> None:
        self._undo: list[Op] = []
        self._redo: list[Op] = []
        self._limit = limit
        self._txn: list[Op] | None = None  # open transaction buffer
        # Monotonic count of every op ever recorded (never trimmed/reset): the
        # Session snapshots it at open/save to detect unsaved edits cheaply.
        self.total_recorded = 0

    # -- recording ----------------------------------------------------------------
    def record(self, op: Op) -> None:
        """Record a completed op. Clears the redo stack (standard semantics)."""
        self.total_recorded += 1
        if self._txn is not None:
            self._txn.append(op)
            return
        self._undo.append(op)
        self._redo.clear()
        self._trim()

    def begin(self) -> None:
        """Open a transaction; subsequent records buffer until commit/rollback."""
        self._txn = []

    def commit(self, label: str = "") -> Op | None:
        """Collapse the open transaction into one compound undo entry."""
        if self._txn is None:
            return None
        children, self._txn = self._txn, None
        if not children:
            return None
        compound = Op(command="chain", label=label or f"{len(children)} ops", children=children)
        self._undo.append(compound)
        self._redo.clear()
        self._trim()
        return compound

    def rollback(self) -> list[Op]:
        """Abort the open transaction; return its ops (newest first) for inverse-apply."""
        children, self._txn = (self._txn or []), None
        return list(reversed(children))

    def snapshot_state(self) -> tuple[list[Op], list[Op], list[Op] | None, int]:
        """Capture exact bookkeeping state for a caller-managed atomic operation."""

        return (
            list(self._undo),
            list(self._redo),
            None if self._txn is None else list(self._txn),
            self.total_recorded,
        )

    def restore_state(
        self, state: tuple[list[Op], list[Op], list[Op] | None, int]
    ) -> None:
        """Restore a checkpoint after its associated external operation failed."""

        undo, redo, transaction, total_recorded = state
        self._undo[:] = undo
        self._redo[:] = redo
        self._txn = None if transaction is None else list(transaction)
        self.total_recorded = total_recorded

    # -- undo / redo --------------------------------------------------------------
    def peek_undo(self, n: int = 1) -> list[Op]:
        """Return the entries that would be undone without changing either stack."""
        return list(reversed(self._undo[-max(1, n):]))

    def peek_redo(self, n: int = 1) -> list[Op]:
        """Return the entries that would be redone without changing either stack."""
        return list(reversed(self._redo[-max(1, n):]))

    def commit_undo(self, count: int) -> None:
        for _ in range(count):
            self._redo.append(self._undo.pop())
        if count:
            # an applied undo CHANGES the stage relative to the last save; the
            # Session calls this directly (not undo()), so the unsaved-edit
            # counter must advance HERE or `transform; save; undo; open`
            # silently discards the undo
            self.total_recorded += 1

    def commit_redo(self, count: int) -> None:
        for _ in range(count):
            self._undo.append(self._redo.pop())
        if count:
            self.total_recorded += 1  # same divergence logic as commit_undo

    def undo(self, n: int = 1) -> list[Op]:
        """Pop up to n entries onto the redo stack; return them for the Session to apply."""
        popped = self.peek_undo(n)
        self.commit_undo(len(popped))  # commit_undo advances the unsaved-edit counter
        return popped

    def redo(self, n: int = 1) -> list[Op]:
        popped = self.peek_redo(n)
        self.commit_redo(len(popped))  # commit_redo advances the unsaved-edit counter
        return popped

    # -- introspection ------------------------------------------------------------
    def entries(self, n: int | None = None) -> list[Op]:
        """Most-recent-first view of the undo stack (PR-7.3)."""
        items = list(reversed(self._undo))
        return items[:n] if n else items

    def reset_baseline(self) -> None:
        """Clear both stacks — e.g. after `checkpoint load` (state-model.md §4)."""
        self._undo.clear()
        self._redo.clear()
        self._txn = None

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def _trim(self) -> None:
        if len(self._undo) > self._limit:
            del self._undo[: len(self._undo) - self._limit]
