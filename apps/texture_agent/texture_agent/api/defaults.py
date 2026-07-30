# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central defaults for Texture Agent.

Mirrors apps/material_agent/material_agent/api/defaults.py so operators
have the same mental model across agents.
"""

# ============================================================================
# LLM Defaults (used by auto-prompt generation in GeneratePromptsTask)
# ============================================================================

DEFAULT_LLM_BACKEND: str = "nim"
DEFAULT_LLM_MODEL: str = "google/gemma-4-31b-it"
DEFAULT_LLM_TEMPERATURE: float = 0.7
# Sized for a multi-material JSON response. Verbose instruct models
# (observed with cosmos-reason2-8b during sidecar testing) produce
# ~500 char/material and at 8 materials a 2048-token cap truncates
# mid-JSON, tripping the fenced-block regex in
# extract_json_from_llm_response and falling back to templated prompts.
# 8192 gives comfortable headroom across backends.
DEFAULT_LLM_MAX_TOKENS: int = 8192
