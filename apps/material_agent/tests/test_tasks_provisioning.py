# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Model Provisioning task."""

import os
from unittest.mock import Mock, patch

import pytest

from material_agent.tasks import ModelProvisioningTask


class TestModelProvisioningTask:
    """Tests for model provisioning task."""

    def test_init(self):
        """Test task initialization."""
        task = ModelProvisioningTask()
        assert task.name == "ModelProvisioning"
        assert task.description == "Create VLM and LLM instances from configuration"

    def test_no_config_raises_error(self):
        """Test that missing config raises ValueError."""
        task = ModelProvisioningTask()
        context = {}

        with pytest.raises(
            ValueError,
            match="'config' not found in context",
        ):
            task.run(context, None)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_api_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_create_vlm_only(self, mock_create_vlm):
        """Test creating only VLM when only VLM is configured."""
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                }
            }
        }

        result = task.run(context, None)

        # Verify VLM was created
        assert "vlm" in result
        assert result["vlm"] is mock_vlm
        mock_create_vlm.assert_called_once()

        # Verify other models were not created but keys exist with None values
        assert result["llm"] is None
        assert result["llm_judge"] is None

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_api_key"})
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_create_llm_only(self, mock_create_chat):
        """Test creating only LLM when only LLM is configured."""
        mock_llm = Mock()
        mock_create_chat.return_value = mock_llm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "llm": {
                    "backend": "nim",
                    "model": "test-llm",
                }
            }
        }

        result = task.run(context, None)

        # Verify LLM was created
        assert "llm" in result
        assert result["llm"] is mock_llm
        mock_create_chat.assert_called_once()

        # Verify other models were not created but keys exist with None values
        assert result["vlm"] is None
        assert result["llm_judge"] is None

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_api_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_create_all_models(self, mock_create_chat, mock_create_vlm):
        """Test creating VLM, LLM, and LLM Judge."""
        mock_vlm = Mock()
        mock_llm = Mock()
        mock_judge = Mock()

        mock_create_vlm.return_value = mock_vlm
        mock_create_chat.side_effect = [mock_llm, mock_judge]

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "vlm-model",
                },
                "llm": {
                    "backend": "nim",
                    "model": "llm-model",
                },
                "llm_judge": {
                    "backend": "nim",
                    "model": "judge-model",
                },
            }
        }

        result = task.run(context, None)

        # Verify all models were created
        assert result["vlm"] is mock_vlm
        assert result["llm"] is mock_llm
        assert result["llm_judge"] is mock_judge

        # Verify creation calls
        mock_create_vlm.assert_called_once()
        assert mock_create_chat.call_count == 2

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "secret_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_api_key_from_env(self, mock_create_vlm):
        """Test loading API key from environment variable."""
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                }
            }
        }

        task.run(context, None)

        # Verify API key was passed to create_vlm
        call_kwargs = mock_create_vlm.call_args[1]
        assert call_kwargs["api_key"] == "secret_key"

    @patch.dict(os.environ, {"MA_NIM_API_KEY": "not-used"}, clear=True)
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_local_nim_vlm_base_url_uses_placeholder_api_key(self, mock_create_vlm):
        """Test local NIM VLM creation without NVIDIA_API_KEY."""
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "local-vlm",
                    "base_url": "http://localhost:8000/v1",
                }
            }
        }

        task.run(context, None)

        call_kwargs = mock_create_vlm.call_args[1]
        assert call_kwargs["api_key"] == "not-used"
        assert call_kwargs["base_url"] == "http://localhost:8000/v1"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "secret_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_nim_vlm_base_url_prefers_nvidia_api_key(self, mock_create_vlm):
        """Test NIM VLM base_url keeps real credentials when available."""
        mock_vlm = Mock()
        mock_create_vlm.return_value = mock_vlm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "hosted-vlm",
                    "base_url": "https://inference-api.nvidia.com/v1",
                }
            }
        }

        task.run(context, None)

        call_kwargs = mock_create_vlm.call_args[1]
        assert call_kwargs["api_key"] == "secret_key"
        assert call_kwargs["base_url"] == "https://inference-api.nvidia.com/v1"

    @patch.dict(os.environ, {"MA_NIM_API_KEY": "not-used"}, clear=True)
    def test_hosted_nim_vlm_rejects_placeholder_api_key(self):
        """Test hosted NIM VLM creation rejects local placeholder credentials."""
        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "hosted-vlm",
                }
            }
        }

        with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
            task.run(context, None)

    @patch.dict(os.environ, {"MA_NIM_API_KEY": "not-used"}, clear=True)
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_local_nim_llm_base_url_uses_placeholder_api_key(self, mock_create_chat):
        """Test local NIM LLM creation without NVIDIA_API_KEY."""
        mock_llm = Mock()
        mock_create_chat.return_value = mock_llm

        task = ModelProvisioningTask()
        context = {
            "config": {
                "llm": {
                    "backend": "nim",
                    "model": "local-llm",
                    "base_url": "http://localhost:8001/v1",
                }
            }
        }

        task.run(context, None)

        call_kwargs = mock_create_chat.call_args[1]
        assert call_kwargs["api_key"] == "not-used"
        assert call_kwargs["base_url"] == "http://localhost:8001/v1"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_api_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_vlm_creation_error(self, mock_create_vlm):
        """Test handling VLM creation error."""
        mock_create_vlm.side_effect = Exception("VLM creation failed")

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                }
            }
        }

        with pytest.raises(Exception, match="VLM creation failed"):
            task.run(context, None)

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_api_key"})
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_llm_creation_error(self, mock_create_chat):
        """Test that LLM creation errors are handled gracefully (LLM is optional)."""
        mock_create_chat.side_effect = Exception("LLM creation failed")

        task = ModelProvisioningTask()
        context = {
            "config": {
                "llm": {
                    "backend": "nim",
                    "model": "test-model",
                }
            }
        }

        # Should not raise, as LLM is optional
        result = task.run(context, None)

        # Verify LLM is None since creation failed
        assert result["llm"] is None
        # Verify the rest of the context was still set up
        assert result["vlm"] is None
        assert result["llm_judge"] is None

    def test_empty_config(self):
        """Test with empty config (no models configured)."""
        task = ModelProvisioningTask()
        context = {"config": {}}

        result = task.run(context, None)

        # Should complete but not create any models - keys exist with None values
        assert result["vlm"] is None
        assert result["llm"] is None
        assert result["llm_judge"] is None


class TestVlmInvokeKwargsExtraction:
    """Tests for vlm_invoke_kwargs extraction and context updates."""

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_includes_inference_time_keys(self, mock_create_vlm):
        """Inference-time keys present in VLM config are extracted."""
        mock_create_vlm.return_value = Mock()

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "top_p": 0.9,
                }
            }
        }

        result = task.run(context, None)

        assert result["vlm_invoke_kwargs"] == {
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 0.9,
        }

    @patch.dict(
        os.environ,
        {"NVIDIA_API_KEY": "hosted_key", "MA_NIM_API_KEY": "local_nim_key"},
    )
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_excludes_non_inference_keys(self, mock_create_vlm):
        """Non-inference keys (backend, model, etc.) are excluded."""
        mock_create_vlm.return_value = Mock()

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                    "base_url": "http://localhost:8000",
                    "temperature": 0.5,
                }
            }
        }

        result = task.run(context, None)

        assert "backend" not in result["vlm_invoke_kwargs"]
        assert "model" not in result["vlm_invoke_kwargs"]
        assert "base_url" not in result["vlm_invoke_kwargs"]
        assert result["vlm_invoke_kwargs"] == {"temperature": 0.5}
        assert mock_create_vlm.call_args.kwargs["api_key"] == "local_nim_key"

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_filters_none_values(self, mock_create_vlm):
        """Inference-time keys with None values are excluded."""
        mock_create_vlm.return_value = Mock()

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                    "temperature": 0.7,
                    "max_tokens": None,
                    "top_p": None,
                    "reasoning_effort": "high",
                }
            }
        }

        result = task.run(context, None)

        assert result["vlm_invoke_kwargs"] == {
            "temperature": 0.7,
            "reasoning_effort": "high",
        }
        assert "max_tokens" not in result["vlm_invoke_kwargs"]
        assert "top_p" not in result["vlm_invoke_kwargs"]

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    def test_zero_is_not_filtered(self, mock_create_vlm):
        """Falsy but non-None values (0, 0.0) are kept."""
        mock_create_vlm.return_value = Mock()

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {
                    "backend": "nim",
                    "model": "test-model",
                    "temperature": 0,
                    "frequency_penalty": 0.0,
                }
            }
        }

        result = task.run(context, None)

        assert result["vlm_invoke_kwargs"] == {
            "temperature": 0,
            "frequency_penalty": 0.0,
        }

    def test_empty_vlm_config_gives_empty_invoke_kwargs(self):
        """No VLM config → vlm_invoke_kwargs is empty dict."""
        task = ModelProvisioningTask()
        context = {"config": {}}

        result = task.run(context, None)

        assert result["vlm_invoke_kwargs"] == {}

    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_context_configs_populated(self, mock_create_chat, mock_create_vlm):
        """All config and model context keys are populated correctly."""
        mock_vlm = Mock()
        mock_llm = Mock()
        mock_vlm_judge = Mock()
        mock_llm_judge = Mock()
        mock_create_vlm.side_effect = [mock_vlm, mock_vlm_judge]
        mock_create_chat.side_effect = [mock_llm, mock_llm_judge]

        vlm_cfg = {"backend": "nim", "model": "vlm-m", "temperature": 0.5}
        llm_cfg = {"backend": "nim", "model": "llm-m"}
        vlm_judge_cfg = {"backend": "nim", "model": "vj-m"}
        llm_judge_cfg = {"backend": "nim", "model": "lj-m"}

        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": vlm_cfg,
                "llm": llm_cfg,
                "vlm_judge": vlm_judge_cfg,
                "llm_judge": llm_judge_cfg,
            }
        }

        result = task.run(context, None)

        # Model instances
        assert result["vlm"] is mock_vlm
        assert result["llm"] is mock_llm
        assert result["vlm_judge"] is mock_vlm_judge
        assert result["llm_judge"] is mock_llm_judge

        # Config dicts
        assert result["vlm_config"] == vlm_cfg
        assert result["llm_config"] == llm_cfg
        assert result["vlm_judge_config"] == vlm_judge_cfg
        assert result["llm_judge_config"] == llm_judge_cfg

        # Invoke kwargs from VLM config
        assert result["vlm_invoke_kwargs"] == {"temperature": 0.5}


class TestModelProvisioningLLMGuard:
    """Tests for the llm config guard (null/string handling)."""

    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    def test_llm_null_skipped(self, mock_create_vlm):
        """When llm config is None, LLM creation is skipped."""
        mock_create_vlm.return_value = Mock()
        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {"backend": "nim", "model": "test"},
                "llm": None,
            }
        }
        result = task.run(context, None)
        assert result["llm"] is None

    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    def test_llm_string_skipped(self, mock_create_vlm):
        """When llm config is a string (not dict), LLM creation is skipped."""
        mock_create_vlm.return_value = Mock()
        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {"backend": "nim", "model": "test"},
                "llm": "some_string",
            }
        }
        result = task.run(context, None)
        assert result["llm"] is None

    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    def test_llm_dict_without_backend_skipped(self, mock_create_vlm):
        """When llm config has no backend key, LLM creation returns None."""
        mock_create_vlm.return_value = Mock()
        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {"backend": "nim", "model": "test"},
                "llm": {"model": "some-model"},  # no backend key
            }
        }
        result = task.run(context, None)
        assert result["llm"] is None

    @patch("world_understanding.agentic.domain_tasks.model_provisioning.create_vlm")
    @patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"})
    def test_llm_empty_dict_skipped(self, mock_create_vlm):
        """When llm config is empty dict, LLM creation is skipped."""
        mock_create_vlm.return_value = Mock()
        task = ModelProvisioningTask()
        context = {
            "config": {
                "vlm": {"backend": "nim", "model": "test"},
                "llm": {},
            }
        }
        result = task.run(context, None)
        assert result["llm"] is None

    @patch.dict(
        os.environ,
        {
            "MA_LLM_NIM_BASE_URL": "http://llm-nim:8000/v1",
            "MA_NIM_API_KEY": "not-used",
        },
        clear=True,
    )
    @patch(
        "world_understanding.agentic.domain_tasks.model_provisioning.create_chat_model"
    )
    def test_llm_judge_honors_runtime_nim_env_override(self, mock_create_chat):
        """``MA_LLM_NIM_BASE_URL`` is applied globally for every LLM call at
        runtime via ``chat_models.create_chat_model_from_config``. Provisioning
        must apply the same override so judge / evaluate paths agree with
        runtime — and so CLI preflight (which also applies the override) does
        not greenlight a config that provisioning would then reject."""
        mock_create_chat.return_value = Mock()

        task = ModelProvisioningTask()
        context = {
            "config": {
                "llm_judge": {
                    "backend": "openai",
                    "model": "example-vlm-model",
                    "api_key": "sk-real-openai-key",
                },
            }
        }

        task.run(context, None)

        # Provisioning saw the env override; the openai api_key was dropped
        # along with backend rewrite, and the call resolves to NIM.
        last_call_kwargs = mock_create_chat.call_args.kwargs
        assert last_call_kwargs["backend"] == "nim"
        assert last_call_kwargs["base_url"] == "http://llm-nim:8000/v1"
        assert last_call_kwargs.get("api_key") != "sk-real-openai-key"
