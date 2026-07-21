# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credential-container regressions for generated durable scene config."""

from pathlib import Path

import yaml

from material_agent.scene.config_gen import _write_durable_config


def test_write_durable_config_omits_signed_url_loaded_from_yaml_set(
    tmp_path: Path,
) -> None:
    secret = "never-persist-material-set-signature"
    config = yaml.safe_load(
        "steps:\n"
        "  predict:\n"
        "    reference_images: !!set\n"
        "      ? 'https://assets.example.test/image.png?"
        f"X-Amz-Signature={secret}'\n"
    )
    config_path = tmp_path / "generated.yaml"

    credential_paths = _write_durable_config(config_path, config)

    persisted_text = config_path.read_text(encoding="utf-8")
    persisted = yaml.safe_load(persisted_text)
    assert credential_paths == ("steps.predict.reference_images",)
    assert persisted == {"steps": {"predict": {}}}
    assert secret not in persisted_text


def test_write_durable_config_omits_mapping_with_signed_url_yaml_key(
    tmp_path: Path,
) -> None:
    secret = "never-persist-material-mapping-key-signature"
    config = yaml.safe_load(
        "steps:\n"
        "  predict:\n"
        "    endpoint_routes:\n"
        "      'https://assets.example.test/object.usd?"
        f"X-Amz-Signature={secret}': primary\n"
        "      public: fallback\n"
    )
    config_path = tmp_path / "generated.yaml"

    credential_paths = _write_durable_config(config_path, config)

    persisted_text = config_path.read_text(encoding="utf-8")
    persisted = yaml.safe_load(persisted_text)
    assert credential_paths == ("steps.predict.endpoint_routes",)
    assert persisted == {"steps": {"predict": {}}}
    assert secret not in persisted_text
