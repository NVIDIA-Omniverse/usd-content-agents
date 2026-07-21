# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-safe command-boundary helpers."""

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

import typer


def sever_cli_exception_graph(
    command: Callable[..., None],
) -> Callable[..., None]:
    """Replace command failures after releasing runtime frames and arguments."""

    @wraps(command)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        unexpected_failure = False
        try:
            command(*args, **kwargs)
            return
        except typer.Exit as error:
            exit_code = error.exit_code
        except Exception:
            exit_code = 1
            unexpected_failure = True

        # Paths and session identifiers are valid raw runtime arguments, but
        # they must not survive in the replacement exception traceback.
        args = ()
        kwargs = {}
        if unexpected_failure:
            # Expected failures should be reported inside the command with a
            # value-free message. Logging happens only after leaving the
            # rejected exception handler and releasing its runtime arguments.
            try:
                logging.getLogger(command.__module__).error("CLI command failed")
            except Exception:
                # A broken logging handler must not restore the rejected
                # exception graph or prevent the safe replacement exit.
                pass
        raise typer.Exit(exit_code) from None

    return wrapped
