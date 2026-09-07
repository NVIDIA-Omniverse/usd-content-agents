# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architectural-invariant tests for Part-1.1 task entry points.

The NL-prompt interpreter (:mod:`physics_agent.tasks.interpret_user_prompt_tuning`)
and the tune judge (:mod:`physics_agent.tasks.judge_tune`) are CLI/REST entry
points. They must NOT pay the import cost of the optimizer stack
(``botorch`` / ``torch`` / ``ovphysx`` / ``physics_agent.tuning.optimizers``)
just to read a prompt or judge an output.

This is a hard architectural invariant per closed issue #51. We assert it by
running each target import in a subprocess with an ``importlib`` meta-path
finder installed *before* the target loads — that finder raises
``ModuleNotFoundError`` for any forbidden name. If the target module imports
successfully under the blocker, the invariant holds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Test file lives at: apps/physics_agent/tests/test_tasks_no_optimizer_imports.py
# parents[0] = tests, [1] = physics_agent, [2] = apps, [3] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Forbidden imports — root-name matched, so e.g. ``botorch.acquisition.foo``
# is also blocked. ``physics_agent.tuning.optimizers`` is the in-repo dispatch
# module that lazily imports botorch; it is matched as a full dotted name.
FORBIDDEN: list[str] = [
    "botorch",
    "torch",
    "ovphysx",
    "physics_agent.tuning.optimizers",
    "world_understanding.optimization.optimizers",
]


def _build_subprocess_source(body: str) -> str:
    """Wrap ``body`` with the import-blocker boilerplate.

    The blocker is installed at ``sys.meta_path[0]`` *before* any target
    import runs. ``body`` is dedented Python source that performs the actual
    target import + assertions. It must print ``OK`` on success.
    """
    return (
        textwrap.dedent(
            """\
        import sys
        from importlib.abc import MetaPathFinder

        FORBIDDEN = {forbidden!r}

        class _Blocker(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".")[0]
                if root in FORBIDDEN or fullname in FORBIDDEN:
                    raise ModuleNotFoundError(
                        f"Forbidden import {{fullname!r}} blocked by test"
                    )
                return None

        # Pre-flight: none of the forbidden modules should already be loaded
        # (the subprocess starts clean, but be defensive against env quirks).
        _preloaded = [m for m in FORBIDDEN if m in sys.modules]
        assert not _preloaded, f"Forbidden modules pre-loaded: {{_preloaded}}"

        sys.meta_path.insert(0, _Blocker())
        """
        ).format(forbidden=FORBIDDEN)
        + "\n"
        + textwrap.dedent(body)
    )


def _run(body: str) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with the blocker installed and ``body`` appended.

    The subprocess inherits ``PYTHONPATH`` so ``physics_agent`` resolves the
    same way it does under pytest (which puts ``apps/physics_agent`` on
    ``sys.path`` via ``[tool.pytest.ini_options].pythonpath``).
    """
    code = _build_subprocess_source(body)

    env = os.environ.copy()
    extra_paths = [
        str(REPO_ROOT),
        str(REPO_ROOT / "apps" / "physics_agent"),
    ]
    existing = env.get("PYTHONPATH", "")
    parts = list(extra_paths)
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
        env=env,
    )


def _assert_clean(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        "Subprocess failed.\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    assert "OK" in result.stdout, (
        f"Expected 'OK' marker in stdout, got:\n{result.stdout}"
    )


# ---------------------------------------------------------------------------
# 1. Pure import — interpreter
# ---------------------------------------------------------------------------


def test_interpreter_imports_without_optimizer_stack() -> None:
    body = """
        __import__("physics_agent.tasks.interpret_user_prompt_tuning")
        leaked = [m for m in FORBIDDEN if m in sys.modules]
        assert not leaked, f"Forbidden modules leaked: {leaked}"
        print("OK")
    """
    _assert_clean(_run(body))


# ---------------------------------------------------------------------------
# 2. Pure import — judge
# ---------------------------------------------------------------------------


def test_judge_imports_without_optimizer_stack() -> None:
    body = """
        __import__("physics_agent.tasks.judge_tune")
        leaked = [m for m in FORBIDDEN if m in sys.modules]
        assert not leaked, f"Forbidden modules leaked: {leaked}"
        print("OK")
    """
    _assert_clean(_run(body))


# ---------------------------------------------------------------------------
# 3. Call-time cleanliness — interpreter
# ---------------------------------------------------------------------------


def test_interpreter_can_call_infer_through_blocker() -> None:
    """Stronger invariant: the interpreter's *call path* (not just import)
    stays clean of the optimizer stack.

    We patch ``generate_chat_response`` on the interpreter module after it
    loads, returning a canned ``drop_settle`` JSON. Then we drive
    :func:`infer_scenario_from_prompt` to a successful Scenario and
    re-check ``sys.modules`` for forbidden leaks.
    """
    body = """
        import json

        # Import target under the blocker.
        from physics_agent.tasks import interpret_user_prompt_tuning as mod
        from physics_agent.tuning.types import Scenario

        # Replace the module-level chat binding with a deterministic stub.
        _payload = {
            "name": "drop_settle",
            "metric": "settle_distance",
            "target": {
                "drop_height_m": 0.5,
                "duration_s": 2.0,
                "gravity": -9.81,
            },
            "parameters": [
                {"name": "restitution", "min": 0.4, "max": 0.95},
                {"name": "mass_scale", "min": 0.7, "max": 1.3},
            ],
        }

        def _fake_chat(*, chat_model, prompt, system_prompt):
            return {"response": json.dumps(_payload)}

        mod.generate_chat_response = _fake_chat

        class _ChatStub:
            model_name = "test-model"

        sc = mod.infer_scenario_from_prompt(
            "make this object bouncy",
            chat_model=_ChatStub(),
        )
        assert isinstance(sc, Scenario), f"got {type(sc).__name__}"
        assert sc.name == "drop_settle"

        leaked = [m for m in FORBIDDEN if m in sys.modules]
        assert not leaked, f"Forbidden modules leaked: {leaked}"
        print("OK")
    """
    _assert_clean(_run(body))


# ---------------------------------------------------------------------------
# 4. Call-time cleanliness — judge with vlm_model=None
# ---------------------------------------------------------------------------


def test_judge_can_run_through_blocker_with_chat_model_none() -> None:
    """The judge with no VLM degrades to programmatic-only,
    returning ``llm_unavailable=True``. Verify the call path imports nothing
    forbidden.
    """
    body = """
        from physics_agent.tasks.judge_tune import (
            JudgeResult,
            run_tune_judge,
        )
        from physics_agent.tuning.types import (
            Scenario,
            TrialRecord,
            TunableParam,
        )

        scenario = Scenario(
            name="drop_settle",
            params=(TunableParam("mass_scale", 0.5, 1.5),),
            target={
                "drop_height_m": 0.5,
                "duration_s": 2.0,
                "gravity": -9.81,
            },
            metric="settle_distance",
        )
        history = [
            TrialRecord(trial_index=0, params={"mass_scale": 1.0}, score=0.1),
            TrialRecord(trial_index=1, params={"mass_scale": 1.2}, score=0.05),
        ]
        best_params = {"mass_scale": 1.2}

        result = run_tune_judge(
            scenario,
            history,
            best_params,
            chat_model=None,
        )
        assert isinstance(result, JudgeResult)
        assert result.llm_unavailable is True
        # With no VLM, llm_score collapses to programmatic_score.
        assert result.llm_score == result.programmatic_score
        # decision is well-formed.
        assert result.decision in ("approve", "continue")

        leaked = [m for m in FORBIDDEN if m in sys.modules]
        assert not leaked, f"Forbidden modules leaked: {leaked}"
        print("OK")
    """
    _assert_clean(_run(body))
