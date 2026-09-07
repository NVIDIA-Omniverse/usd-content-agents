# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the public Geometry Agent service."""

from __future__ import annotations

import socket
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

AuthoringFormat = Literal[
    "step",
    "stl",
    "obj",
    "ply",
    "glb",
    "gltf",
    "3mf",
    "usd",
    "usda",
    "usdc",
]

_RESERVED_AUTHORING_PROVIDER_IDS = frozenset(
    {
        "build123d-http",
        "forgecad-http",
    }
)


class Settings(BaseSettings):
    """Runtime settings loaded from ``GEOMETRY_AGENT_SERVICE_*`` variables."""

    model_config = SettingsConfigDict(
        env_prefix="GEOMETRY_AGENT_SERVICE_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8776, ge=1, le=65535)
    workspace_root: str = "/var/geometry-agent-service/workspace"
    instance_id: str = Field(
        default_factory=socket.gethostname,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    job_owner_heartbeat_seconds: float = Field(default=15.0, gt=0.0)
    job_owner_lease_timeout_seconds: float = Field(default=60.0, gt=0.0)
    api_key: str | None = None
    # The public service image does not bundle a simulation runtime. Deployments
    # that provide one can opt in explicitly.
    default_runtime_engine: Literal["ovphysx", "fake", "none"] = "none"
    render_backend: Literal["ovrtx", "remote"] = "ovrtx"
    render_remote_base_url: str | None = None
    render_remote_api_key: SecretStr | None = None
    render_remote_allow_unauthenticated_identity: bool = False

    delegated_authoring_provider_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    delegated_authoring_provider_label: str = Field(
        default="Delegated authoring worker",
        min_length=1,
        max_length=128,
    )
    delegated_authoring_endpoint_url: str | None = None
    delegated_authoring_bearer_token: SecretStr | None = None
    delegated_authoring_rights_assertion: str | None = None
    delegated_authoring_supported_formats: tuple[AuthoringFormat, ...] = (
        "step",
        "stl",
        "usd",
        "usda",
    )
    delegated_authoring_supports_image: bool = True
    delegated_authoring_supports_revision: bool = True
    delegated_authoring_supports_export: bool = True
    delegated_authoring_returns_native_source: bool = True

    build123d_endpoint_url: str | None = None
    build123d_bearer_token: SecretStr | None = None
    build123d_rights_assertion: str | None = None
    build123d_supported_formats: tuple[AuthoringFormat, ...] = ("step",)
    build123d_supports_text: bool = True
    build123d_supports_image: bool = False
    build123d_supports_revision: bool = False
    build123d_supports_export: bool = False
    build123d_supports_semantic_parameters: bool = False
    build123d_supports_parameter_definitions: bool = False
    build123d_supports_semantic_parts: bool = False
    build123d_supports_provider_assertions: bool = False
    build123d_max_family_variants: int = Field(default=0, ge=0, le=64)

    forgecad_authoring_endpoint_url: str | None = None
    forgecad_authoring_bearer_token: SecretStr | None = None
    forgecad_authoring_rights_assertion: str | None = None
    forgecad_automated_use_authorized: bool = False
    forgecad_authoring_supported_formats: tuple[AuthoringFormat, ...] = ("step",)
    forgecad_authoring_supports_text: bool = True
    forgecad_authoring_supports_image: bool = False
    forgecad_authoring_supports_revision: bool = False
    forgecad_authoring_supports_export: bool = False
    forgecad_authoring_supports_semantic_parameters: bool = False
    forgecad_authoring_supports_parameter_definitions: bool = False
    forgecad_authoring_supports_semantic_parts: bool = False
    forgecad_authoring_supports_provider_assertions: bool = False
    forgecad_authoring_max_family_variants: int = Field(default=0, ge=0, le=64)
    forgecad_authoring_returns_native_source: bool = False
    forgecad_authoring_connect_timeout_seconds: float = Field(
        default=10.0, gt=0.0, le=300.0
    )
    forgecad_authoring_read_timeout_seconds: float = Field(
        default=300.0, ge=1.0, le=900.0
    )

    max_upload_bytes: int = Field(default=128 * 1024 * 1024, ge=1)
    max_reference_image_bytes: int = Field(default=16 * 1024 * 1024, ge=1)
    max_request_bytes: int = Field(default=130 * 1024 * 1024, ge=1)
    max_generated_artifact_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_archive_members: int = Field(default=256, ge=1, le=100_000)
    max_archive_unpacked_bytes: int = Field(
        default=512 * 1024 * 1024,
        ge=1,
    )
    max_archive_compression_ratio: float = Field(default=200.0, ge=1.0)

    @model_validator(mode="after")
    def _validate_limits(self) -> Settings:
        if self.job_owner_heartbeat_seconds >= self.job_owner_lease_timeout_seconds:
            raise ValueError(
                "job_owner_heartbeat_seconds must be less than "
                "job_owner_lease_timeout_seconds"
            )
        if self.max_request_bytes <= self.max_upload_bytes:
            raise ValueError(
                "max_request_bytes must exceed max_upload_bytes to allow multipart framing"
            )
        if self.max_reference_image_bytes > self.max_upload_bytes:
            raise ValueError(
                "max_reference_image_bytes must not exceed max_upload_bytes"
            )
        if self.render_backend == "remote":
            if not self.render_remote_base_url:
                raise ValueError(
                    "render_remote_base_url is required when render_backend is remote"
                )
            if (
                self.render_remote_api_key is None
                or not self.render_remote_api_key.get_secret_value()
            ) and not self.render_remote_allow_unauthenticated_identity:
                raise ValueError(
                    "render_remote_api_key is required when render_backend is remote "
                    "unless render_remote_allow_unauthenticated_identity is enabled"
                )
        if self.render_backend != "remote" and (
            self.render_remote_base_url
            or self.render_remote_api_key is not None
            or self.render_remote_allow_unauthenticated_identity
        ):
            raise ValueError(
                "remote render endpoint settings require render_backend=remote"
            )
        self._validate_worker_configuration(
            name="delegated_authoring",
            endpoint_url=self.delegated_authoring_endpoint_url,
            bearer_token=self.delegated_authoring_bearer_token,
            rights_assertion=self.delegated_authoring_rights_assertion,
            supported_formats=self.delegated_authoring_supported_formats,
        )
        if bool(self.delegated_authoring_provider_id) != bool(
            self.delegated_authoring_endpoint_url
        ):
            raise ValueError(
                "delegated_authoring_provider_id and endpoint_url must be configured together"
            )
        if self.delegated_authoring_provider_id in _RESERVED_AUTHORING_PROVIDER_IDS:
            raise ValueError(
                "delegated_authoring_provider_id must not reuse a built-in provider ID"
            )
        self._validate_worker_configuration(
            name="build123d",
            endpoint_url=self.build123d_endpoint_url,
            bearer_token=self.build123d_bearer_token,
            rights_assertion=self.build123d_rights_assertion,
            supported_formats=self.build123d_supported_formats,
        )
        self._validate_worker_configuration(
            name="forgecad_authoring",
            endpoint_url=self.forgecad_authoring_endpoint_url,
            bearer_token=self.forgecad_authoring_bearer_token,
            rights_assertion=self.forgecad_authoring_rights_assertion,
            supported_formats=self.forgecad_authoring_supported_formats,
        )
        if (
            self.forgecad_authoring_endpoint_url
            and not self.forgecad_automated_use_authorized
        ):
            raise ValueError(
                "forgecad_automated_use_authorized must be true when ForgeCAD "
                "authoring is configured"
            )
        if (
            self.forgecad_automated_use_authorized
            and not self.forgecad_authoring_endpoint_url
        ):
            raise ValueError(
                "forgecad_authoring_endpoint_url is required when automated use is authorized"
            )
        if not (
            self.forgecad_authoring_supports_text
            or self.forgecad_authoring_supports_image
        ):
            raise ValueError("ForgeCAD authoring must support text or image input")
        if (
            self.forgecad_authoring_supports_parameter_definitions
            and not self.forgecad_authoring_supports_semantic_parameters
        ):
            raise ValueError(
                "ForgeCAD parameter definitions require semantic parameter support"
            )
        if self.forgecad_authoring_max_family_variants and not (
            self.forgecad_authoring_supports_revision
            and self.forgecad_authoring_supports_parameter_definitions
        ):
            raise ValueError(
                "ForgeCAD parameter families require revision and parameter definitions"
            )
        if (
            self.forgecad_authoring_connect_timeout_seconds
            > self.forgecad_authoring_read_timeout_seconds
        ):
            raise ValueError(
                "ForgeCAD connect timeout must not exceed its read timeout"
            )
        return self

    @staticmethod
    def _validate_worker_configuration(
        *,
        name: str,
        endpoint_url: str | None,
        bearer_token: SecretStr | None,
        rights_assertion: str | None,
        supported_formats: tuple[AuthoringFormat, ...],
    ) -> None:
        if not supported_formats or len(supported_formats) != len(
            set(supported_formats)
        ):
            raise ValueError(
                f"{name}_supported_formats must be a non-empty unique tuple"
            )
        if endpoint_url and not rights_assertion:
            raise ValueError(f"{name}_rights_assertion is required when configured")
        if (
            bearer_token is not None or rights_assertion is not None
        ) and not endpoint_url:
            raise ValueError(
                f"{name}_endpoint_url is required for worker credentials or rights"
            )


settings = Settings()
