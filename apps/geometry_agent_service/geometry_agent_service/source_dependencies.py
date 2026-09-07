# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for shared geometry source dependency validation."""

from geometry_authoring_contracts.source_dependencies import (
    SourceDependencyError,
    validate_3mf_package_dependencies,
    validate_materialized_source_dependencies,
)

__all__ = [
    "SourceDependencyError",
    "validate_3mf_package_dependencies",
    "validate_materialized_source_dependencies",
]
