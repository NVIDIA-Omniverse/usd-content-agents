# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Domain-neutral control flow for iterative refinement."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RefinementIteration[StateT]:
    """One evaluate, judge, and optionally revise cycle."""

    iteration: int
    max_iterations: int
    state: StateT

    @property
    def is_last(self) -> bool:
        """Whether this is the final allowed iteration."""

        return self.iteration == self.max_iterations


class RefinementLoop[StateT]:
    """Own iteration limits and state promotion for a refinement workflow.

    Domain code evaluates the current state, gathers evidence, and judges the
    result. It then calls :meth:`approve`, :meth:`continue_with`, or
    :meth:`stop`. The next iteration cannot begin until that transition is
    explicit.
    """

    def __init__(self, *, initial_state: StateT, max_iterations: int) -> None:
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations <= 0
        ):
            raise ValueError("max_iterations must be positive")
        self._state = initial_state
        self._max_iterations = max_iterations
        self._next_iteration = 1
        self._active_iteration: int | None = None
        self._termination_reason: str | None = None

    @property
    def termination_reason(self) -> str | None:
        """Why the loop ended, or ``None`` while it can still run."""

        return self._termination_reason

    @property
    def is_complete(self) -> bool:
        """Whether the loop has reached a terminal decision."""

        return self._termination_reason is not None

    @property
    def state(self) -> StateT:
        """The most recently accepted state."""

        return self._state

    def begin_iteration(self) -> RefinementIteration[StateT] | None:
        """Begin the next iteration, or return ``None`` after termination."""

        if self.is_complete:
            return None
        if self._active_iteration is not None:
            raise RuntimeError(
                "active refinement iteration requires an explicit transition"
            )
        self._active_iteration = self._next_iteration
        return RefinementIteration(
            iteration=self._active_iteration,
            max_iterations=self._max_iterations,
            state=self._state,
        )

    def continue_with(self, next_state: StateT) -> None:
        """Promote a revised state, or terminate when the cap was reached."""

        iteration = self._require_active_iteration()
        self._active_iteration = None
        if iteration == self._max_iterations:
            self._termination_reason = "max_iterations"
            return
        self._state = next_state
        self._next_iteration = iteration + 1

    def approve(self) -> None:
        """Accept the active iteration and terminate successfully."""

        self.stop("approved")

    def stop(self, reason: str) -> None:
        """Terminate the loop for a domain-defined reason."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("refinement stop reason must not be empty")
        if self.is_complete:
            raise RuntimeError("refinement loop is already complete")
        self._active_iteration = None
        self._termination_reason = reason

    def _require_active_iteration(self) -> int:
        if self.is_complete:
            raise RuntimeError("refinement loop is already complete")
        if self._active_iteration is None:
            raise RuntimeError("no active refinement iteration")
        return self._active_iteration


@dataclass(frozen=True)
class RefinementDecision:
    """A domain policy decision after one evaluate-and-judge cycle."""

    action: Literal["approve", "continue", "stop"]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.action not in {"approve", "continue", "stop"}:
            raise ValueError(f"unsupported refinement action: {self.action!r}")
        if self.action == "stop":
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("stop decisions require a non-empty reason")
        elif self.reason is not None:
            raise ValueError(f"{self.action} decisions must not provide a reason")

    @classmethod
    def approve(cls) -> RefinementDecision:
        """Approve the current state."""

        return cls("approve")

    @classmethod
    def continue_(cls) -> RefinementDecision:
        """Request a revised state and another iteration."""

        return cls("continue")

    @classmethod
    def stop(cls, reason: str) -> RefinementDecision:
        """Stop with a domain-defined terminal reason."""

        return cls("stop", reason)


@dataclass(frozen=True)
class RefinementRecord[StateT, EvaluationT, JudgmentT]:
    """One completed evaluate, judge, and transition cycle."""

    iteration: RefinementIteration[StateT]
    evaluation: EvaluationT
    judgment: JudgmentT
    decision: RefinementDecision
    next_state: StateT | None = None


@dataclass
class RefinementRun[StateT, EvaluationT, JudgmentT]:
    """Result of a callback-driven refinement workflow."""

    records: list[RefinementRecord[StateT, EvaluationT, JudgmentT]] = field(
        default_factory=list
    )
    final_state: StateT | None = None
    termination_reason: str = ""
    cancelled: bool = False
    error: Exception | None = None

    @property
    def approved(self) -> bool:
        """Whether a domain judge approved the final state."""

        return self.termination_reason == "approved"


def run_refinement[StateT, EvaluationT, JudgmentT](
    *,
    initial_state: StateT,
    max_iterations: int,
    evaluate: Callable[[RefinementIteration[StateT]], EvaluationT],
    judge: Callable[[RefinementIteration[StateT], EvaluationT], JudgmentT],
    decide: Callable[
        [RefinementIteration[StateT], EvaluationT, JudgmentT], RefinementDecision
    ],
    revise: Callable[[RefinementIteration[StateT], EvaluationT, JudgmentT], StateT],
    cancel_check: Callable[[], bool] | None = None,
    on_iteration: Callable[[RefinementRecord[StateT, EvaluationT, JudgmentT]], None]
    | None = None,
    on_complete: Callable[[RefinementRun[StateT, EvaluationT, JudgmentT]], None]
    | None = None,
    caught_exceptions: tuple[type[Exception], ...] = (),
    cancellation_reason: str = "cancelled",
    error_reason: str = "error",
) -> RefinementRun[StateT, EvaluationT, JudgmentT]:
    """Run evaluate, judge, decide, and revise callbacks until termination.

    The shared engine owns iteration limits, state promotion, cancellation, and
    transition records. Domain adapters own evaluation, evidence, judgment,
    revision, and artifact side effects. Exceptions propagate by default;
    callers that need result-shaped failures can opt into ``caught_exceptions``.
    """

    loop = RefinementLoop(initial_state=initial_state, max_iterations=max_iterations)
    records: list[RefinementRecord[StateT, EvaluationT, JudgmentT]] = []
    caught_error: Exception | None = None

    callbacks = {
        "evaluate": evaluate,
        "judge": judge,
        "decide": decide,
        "revise": revise,
    }
    if cancel_check is not None:
        callbacks["cancel_check"] = cancel_check
    if on_iteration is not None:
        callbacks["on_iteration"] = on_iteration
    if on_complete is not None:
        callbacks["on_complete"] = on_complete
    for name, callback in callbacks.items():
        if not callable(callback):
            raise TypeError(f"{name} must be callable")
    if not isinstance(caught_exceptions, tuple) or any(
        not isinstance(error_type, type) or not issubclass(error_type, Exception)
        for error_type in caught_exceptions
    ):
        raise TypeError("caught_exceptions must contain Exception classes")
    for name, reason in (
        ("cancellation_reason", cancellation_reason),
        ("error_reason", error_reason),
    ):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{name} must not be empty")

    def cancellation_requested() -> bool:
        return cancel_check is not None and cancel_check()

    try:
        while (iteration := loop.begin_iteration()) is not None:
            if cancellation_requested():
                loop.stop(cancellation_reason)
                break

            evaluation = evaluate(iteration)
            if cancellation_requested():
                loop.stop(cancellation_reason)
                break

            judgment = judge(iteration, evaluation)
            if cancellation_requested():
                loop.stop(cancellation_reason)
                break

            decision = decide(iteration, evaluation, judgment)
            if not isinstance(decision, RefinementDecision):
                raise TypeError("decide must return RefinementDecision")

            if cancellation_requested():
                decision = RefinementDecision.stop(cancellation_reason)
            next_state: StateT | None = None
            transition_state = iteration.state
            if decision.action == "continue" and not iteration.is_last:
                revised_state = revise(iteration, evaluation, judgment)
                next_state = revised_state
                transition_state = revised_state

            record = RefinementRecord(
                iteration=iteration,
                evaluation=evaluation,
                judgment=judgment,
                decision=decision,
                next_state=next_state,
            )
            records.append(record)
            if on_iteration is not None:
                on_iteration(record)

            if decision.action == "approve":
                loop.approve()
            elif decision.action == "stop":
                assert decision.reason is not None
                loop.stop(decision.reason)
            else:
                loop.continue_with(transition_state)
    except caught_exceptions as error:
        caught_error = error
        if not loop.is_complete:
            loop.stop(error_reason)

    termination_reason = loop.termination_reason
    if termination_reason is None:
        raise RuntimeError("refinement workflow ended without a terminal decision")
    result = RefinementRun(
        records=records,
        final_state=loop.state,
        termination_reason=termination_reason,
        cancelled=termination_reason == cancellation_reason,
        error=caught_error,
    )
    if on_complete is not None:
        on_complete(result)
    return result


__all__ = [
    "RefinementDecision",
    "RefinementIteration",
    "RefinementLoop",
    "RefinementRecord",
    "RefinementRun",
    "run_refinement",
]
