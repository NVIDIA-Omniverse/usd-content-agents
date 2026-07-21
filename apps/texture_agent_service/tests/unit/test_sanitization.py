# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``service.sanitization``.

Covers two public response leak surfaces:

- NVCF function-invocation URLs (``https://<id>.invocation.api.nvcf.nvidia.com/...``)
  embedded in error messages by ``str(httpx.HTTPStatusError)``.
- Absolute session-storage paths (``/var/texture-agent/sessions/<sid>/...``).
"""

from __future__ import annotations

import pytest

from ...service.sanitization import sanitize_message, sanitize_step_stats


class TestSanitizeMessage:
    def test_redacts_nvcf_invocation_url_with_path(self) -> None:
        msg = (
            "NVCF request failed with HTTP 500: Server error '500 Internal "
            "Server Error' for url 'https://abc12345-def6-4789-9abc-def012345678."
            "invocation.api.nvcf.nvidia.com/v2/nvcf/exec/functions/abc/versions/v1'"
        )
        out = sanitize_message(msg)
        assert "<nvcf-endpoint>" in out
        assert "nvcf.nvidia.com" not in out
        assert "abc12345" not in out
        # Surrounding diagnostic text is preserved.
        assert "HTTP 500" in out

    def test_redacts_nvcf_status_api_url(self) -> None:
        msg = "Polling failed via https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/req-xyz"
        out = sanitize_message(msg)
        assert "<nvcf-endpoint>" in out
        assert "api.nvcf.nvidia.com" not in out

    def test_redacts_multiple_urls_in_one_message(self) -> None:
        msg = (
            "primary https://aaa.invocation.api.nvcf.nvidia.com/exec failed; "
            "fallback https://bbb.invocation.api.nvcf.nvidia.com/exec also failed"
        )
        out = sanitize_message(msg)
        assert out.count("<nvcf-endpoint>") == 2
        assert "nvcf.nvidia.com" not in out

    def test_redacts_var_texture_agent_paths_unconditionally(self) -> None:
        # Even without an explicit storage root, the docker default is stripped.
        msg = "FileNotFoundError: /var/texture-agent/sessions/abc/input/scene.usd"
        out = sanitize_message(msg)
        assert "<session>" in out
        assert "/var/texture-agent/sessions" not in out

    def test_redacts_custom_storage_root(self) -> None:
        msg = "USD missing at /tmp/dev-sessions/sess-1/cache/output/textured_output.usd"
        out = sanitize_message(msg, storage_root="/tmp/dev-sessions")
        assert "<session>" in out
        assert "/tmp/dev-sessions" not in out
        # Filename suffix is also stripped (the whole tail under root).
        assert "textured_output.usd" not in out

    def test_default_root_still_applied_when_custom_root_passed(self) -> None:
        # Both the configured root AND the docker default get stripped, since
        # the docker image bakes the latter even when local dev uses another.
        msg = (
            "primary /tmp/sess-root/abc/x.usd; "
            "secondary /var/texture-agent/sessions/def/y.usd"
        )
        out = sanitize_message(msg, storage_root="/tmp/sess-root")
        assert out.count("<session>") == 2
        assert "/tmp/sess-root" not in out
        assert "/var/texture-agent/sessions" not in out

    def test_storage_root_with_trailing_slash_is_normalized(self) -> None:
        msg = "missing /tmp/storage/abc/scene.usd"
        out = sanitize_message(msg, storage_root="/tmp/storage/")
        assert "<session>" in out
        assert "/tmp/storage" not in out

    def test_empty_message_passes_through(self) -> None:
        assert sanitize_message("") == ""

    @pytest.mark.parametrize(
        "msg",
        [
            "ordinary error text without any URLs or paths",
            "HTTP 403 Forbidden",
            "Connection reset by peer",
        ],
    )
    def test_clean_messages_pass_through_unchanged(self, msg: str) -> None:
        assert sanitize_message(msg, storage_root="/tmp/x") == msg


class TestSanitizeStepStats:
    def test_none_passes_through(self) -> None:
        assert sanitize_step_stats(None) is None

    def test_per_step_errors_list(self) -> None:
        # Shape produced by ``_extract_step_stats`` for generate/blend.
        stats = {
            "textures_generated": 1,
            "textures_failed": 2,
            "errors": [
                {
                    "material": "Steel",
                    "type": "HTTPStatusError",
                    "status": 500,
                    "message": (
                        "NVCF request failed with HTTP 500: Server error for url "
                        "'https://func1.invocation.api.nvcf.nvidia.com/exec'"
                    ),
                },
                {
                    "material": "Wood",
                    "type": "RuntimeError",
                    "status": None,
                    "message": "no albedo at /var/texture-agent/sessions/sid/cache/textures/wood_albedo.png",
                },
            ],
        }
        out = sanitize_step_stats(stats)
        assert out is not None
        assert "<nvcf-endpoint>" in out["errors"][0]["message"]
        assert "nvcf.nvidia.com" not in out["errors"][0]["message"]
        assert "<session>" in out["errors"][1]["message"]
        assert "/var/texture-agent" not in out["errors"][1]["message"]
        # Non-string fields untouched, type/status preserved for diagnostics.
        assert out["errors"][0]["status"] == 500
        assert out["textures_failed"] == 2

    def test_final_stats_errors_dict_of_lists(self) -> None:
        # Shape produced by ``_extract_final_stats`` after threshold gate.
        stats = {
            "materials_found": 4,
            "errors": {
                "generate_textures": {
                    "count": 2,
                    "errors": [
                        {
                            "material": "A",
                            "message": "https://a1.invocation.api.nvcf.nvidia.com",
                        }
                    ],
                },
                "blend_textures": {
                    "count": 1,
                    "errors": [
                        {
                            "material": "B",
                            "message": "https://b1.invocation.api.nvcf.nvidia.com",
                        }
                    ],
                },
            },
        }
        out = sanitize_step_stats(stats)
        assert out is not None
        gen_msg = out["errors"]["generate_textures"]["errors"][0]["message"]
        blend_msg = out["errors"]["blend_textures"]["errors"][0]["message"]
        assert "<nvcf-endpoint>" in gen_msg
        assert "<nvcf-endpoint>" in blend_msg
        assert "nvcf.nvidia.com" not in gen_msg
        assert "nvcf.nvidia.com" not in blend_msg

    def test_input_is_not_mutated(self) -> None:
        original = {
            "errors": [
                {
                    "material": "A",
                    "message": "https://x.invocation.api.nvcf.nvidia.com/exec",
                }
            ]
        }
        sanitize_step_stats(original)
        assert "x.invocation.api.nvcf.nvidia.com" in original["errors"][0]["message"]
