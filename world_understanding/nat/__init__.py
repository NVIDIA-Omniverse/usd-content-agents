# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NAT adapter modules."""

# Optional imports for runtime loader functionality
try:
    from .runtime_loader import (
        NATWorkflow,
        query_workflow,
        validate_nat_config,
    )

    __all__ = ["NATWorkflow", "query_workflow", "validate_nat_config"]
except ImportError:
    # NAT runtime is optional
    __all__ = []
