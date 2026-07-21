# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session management."""

import json

import pytest

from world_understanding.agentic.session import SessionManager


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_create_new_session(self, tmp_path):
        """Test creating a new session with auto-generated ID."""
        session = SessionManager.create(base_dir=tmp_path, project_name="test_project")

        # Verify session was created
        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
        assert session.project_name == "test_project"
        assert session.session_dir.exists()
        assert session.session_dir.parent == tmp_path
        assert session.session_dir.name.startswith(".")

        # Verify metadata
        assert session.metadata["session_id"] == session.session_id
        assert session.metadata["project_name"] == "test_project"
        assert "created_at" in session.metadata

    def test_create_session_with_custom_id(self, tmp_path):
        """Test creating a session with a specific ID."""
        custom_id = "my-custom-session-123"

        session = SessionManager.create(
            base_dir=tmp_path, project_name="test_project", session_id=custom_id
        )

        assert session.session_id == custom_id
        assert session.session_dir == tmp_path / f".{custom_id}"
        assert session.session_dir.exists()

    def test_from_id_existing_session(self, tmp_path):
        """Test loading an existing session by ID."""
        # Create a session first
        original = SessionManager.create(
            base_dir=tmp_path,
            project_name="test_project",
            session_id="test-session-456",
        )
        original.save_metadata()

        # Load it by ID
        loaded = SessionManager.from_id(
            session_id="test-session-456", base_dir=tmp_path
        )

        assert loaded.session_id == original.session_id
        assert loaded.session_dir == original.session_dir
        assert loaded.metadata["session_id"] == original.session_id

    def test_from_id_nonexistent_session(self, tmp_path):
        """Test loading a nonexistent session raises error."""
        with pytest.raises(FileNotFoundError, match="Session directory not found"):
            SessionManager.from_id(session_id="nonexistent-session", base_dir=tmp_path)

    def test_get_subdir_creates_directory(self, tmp_path):
        """Test get_subdir creates subdirectories."""
        session = SessionManager.create(base_dir=tmp_path)

        dataset_dir = session.get_subdir("dataset")
        assert dataset_dir.exists()
        assert dataset_dir.parent == session.session_dir
        assert dataset_dir.name == "dataset"

        # Test nested subdirectory
        iter_dir = session.get_subdir("iterations/iteration_1")
        assert iter_dir.exists()
        assert iter_dir.name == "iteration_1"
        assert iter_dir.parent.name == "iterations"

    def test_get_subdir_no_create(self, tmp_path):
        """Test get_subdir with create=False."""
        session = SessionManager.create(base_dir=tmp_path)

        # Get subdirectory without creating
        output_dir = session.get_subdir("output", create=False)
        assert not output_dir.exists()
        assert output_dir.parent == session.session_dir

    def test_get_file(self, tmp_path):
        """Test get_file returns correct path."""
        session = SessionManager.create(base_dir=tmp_path)

        config_file = session.get_file("config.yaml")
        assert config_file.parent == session.session_dir
        assert config_file.name == "config.yaml"

        # Test nested file path
        output_file = session.get_file("output/result.json")
        assert output_file.name == "result.json"
        assert output_file.parent.name == "output"

    def test_save_and_load_metadata(self, tmp_path):
        """Test saving and loading session metadata."""
        # Create session with custom metadata
        session = SessionManager.create(
            base_dir=tmp_path,
            project_name="test_project",
            metadata={"custom_field": "custom_value"},
        )

        # Save metadata
        session.save_metadata()

        # Verify metadata file exists
        metadata_file = session.session_dir / ".metadata.json"
        assert metadata_file.exists()

        # Load and verify contents
        with open(metadata_file, encoding="utf-8") as f:
            saved_metadata = json.load(f)

        assert saved_metadata["session_id"] == session.session_id
        assert saved_metadata["project_name"] == "test_project"
        assert saved_metadata["custom_field"] == "custom_value"

        # Load session from ID and verify metadata
        loaded_session = SessionManager.from_id(
            session_id=session.session_id, base_dir=tmp_path
        )
        assert loaded_session.metadata["custom_field"] == "custom_value"

    def test_update_metadata(self, tmp_path):
        """Test updating session metadata."""
        session = SessionManager.create(base_dir=tmp_path)

        # Update metadata
        session.update_metadata(status="running", num_predictions=42, score=0.95)

        # Verify metadata was updated
        assert session.metadata["status"] == "running"
        assert session.metadata["num_predictions"] == 42
        assert session.metadata["score"] == 0.95

        # Verify it was saved to disk
        metadata_file = session.session_dir / ".metadata.json"
        assert metadata_file.exists()

        with open(metadata_file, encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["status"] == "running"
        assert saved["num_predictions"] == 42

    def test_list_sessions_empty(self, tmp_path):
        """Test listing sessions in empty directory."""
        sessions = SessionManager.list_sessions(tmp_path)
        assert sessions == []

    def test_list_sessions_multiple(self, tmp_path):
        """Test listing multiple sessions."""
        # Create multiple sessions
        session1 = SessionManager.create(
            base_dir=tmp_path, project_name="project_a", session_id="session-001"
        )
        session1.save_metadata()

        session2 = SessionManager.create(
            base_dir=tmp_path, project_name="project_b", session_id="session-002"
        )
        session2.save_metadata()

        # List sessions
        sessions = SessionManager.list_sessions(tmp_path)

        assert len(sessions) == 2
        session_ids = {s["session_id"] for s in sessions}
        assert "session-001" in session_ids
        assert "session-002" in session_ids

        # Verify metadata is included
        project_names = {s["project_name"] for s in sessions}
        assert "project_a" in project_names
        assert "project_b" in project_names

        by_id = {item["session_id"]: item for item in sessions}
        assert by_id["session-001"]["session_dir"] == str(session1.session_dir)
        assert by_id["session-002"]["session_dir"] == str(session2.session_dir)

    def test_list_sessions_redacts_unsafe_derived_fields(self, tmp_path):
        """Credential-bearing directory fields never escape through the result."""
        secret = "list-session-result-sentinel-727"
        session_dir = tmp_path / f".artifact?X-Amz-Signature={secret}"
        session_dir.mkdir()
        (session_dir / ".metadata.json").write_text(
            json.dumps(
                {
                    "project_name": f"https://user:{secret}@project.test/name",
                    "artifact": (
                        f"https://assets.test/model.usd?X-Amz-Signature={secret}"
                    ),
                    "safe": "retained",
                    "created_at": "2026-07-16T00:00:00",
                }
            ),
            encoding="utf-8",
        )

        sessions = SessionManager.list_sessions(tmp_path)

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "<redacted>"
        assert sessions[0]["session_dir"] == "<redacted>"
        assert sessions[0]["project_name"] == "<redacted>"
        assert sessions[0]["metadata"]["artifact"] == "<redacted>"
        assert sessions[0]["metadata"]["safe"] == "retained"
        assert secret not in repr(sessions)

    def test_list_sessions_redacts_unsafe_base_path_only(self, tmp_path):
        """A sensitive base path does not alter an otherwise benign session ID."""
        secret = "list-session-base-sentinel-727"
        base_dir = tmp_path / f"runs?X-Amz-Signature={secret}"
        (base_dir / ".ordinary-id").mkdir(parents=True)

        sessions = SessionManager.list_sessions(base_dir)

        assert sessions[0]["session_id"] == "ordinary-id"
        assert sessions[0]["session_dir"] == "<redacted>"
        assert secret not in repr(sessions)

    def test_list_sessions_skips_directory_and_metadata_symlinks(self, tmp_path):
        """Listing never follows aliases to external session data."""
        secret = "list-session-symlink-sentinel-727"
        base_dir = tmp_path / "sessions"
        base_dir.mkdir()
        external_session = tmp_path / "external-session"
        external_session.mkdir()
        (external_session / ".metadata.json").write_text(
            json.dumps({"project_name": f"https://user:{secret}@project.test"}),
            encoding="utf-8",
        )
        (base_dir / ".external-alias").symlink_to(
            external_session,
            target_is_directory=True,
        )

        real_session = base_dir / ".real"
        real_session.mkdir()
        external_metadata = tmp_path / "external-metadata.json"
        external_metadata.write_text(
            json.dumps({"project_name": f"https://user:{secret}@project.test"}),
            encoding="utf-8",
        )
        (real_session / ".metadata.json").symlink_to(external_metadata)

        sessions = SessionManager.list_sessions(base_dir)

        assert sessions == [
            {
                "session_id": "real",
                "session_dir": str(real_session),
                "project_name": "unknown",
                "created_at": None,
                "metadata": {},
            }
        ]
        assert secret not in repr(sessions)

    def test_list_sessions_normalizes_malformed_summary_fields(self, tmp_path):
        """Malformed JSON metadata stays projected and cannot break sorting."""
        valid = tmp_path / ".valid"
        valid.mkdir()
        (valid / ".metadata.json").write_text(
            json.dumps(
                {
                    "project_name": "valid-project",
                    "created_at": "2026-07-16T00:00:00",
                }
            ),
            encoding="utf-8",
        )
        malformed = tmp_path / ".malformed"
        malformed.mkdir()
        malformed_metadata = {
            "project_name": ["not", "a", "string"],
            "created_at": {"not": "a string"},
            "nested": {"safe": [1, True, None]},
        }
        (malformed / ".metadata.json").write_text(
            json.dumps(malformed_metadata),
            encoding="utf-8",
        )
        non_mapping = tmp_path / ".non-mapping"
        non_mapping.mkdir()
        (non_mapping / ".metadata.json").write_text("[]", encoding="utf-8")

        sessions = SessionManager.list_sessions(tmp_path)

        assert sessions[0]["session_id"] == "valid"
        by_id = {item["session_id"]: item for item in sessions}
        assert by_id["malformed"]["project_name"] == "unknown"
        assert by_id["malformed"]["created_at"] is None
        assert by_id["malformed"]["metadata"] == malformed_metadata
        assert by_id["non-mapping"]["project_name"] == "unknown"
        assert by_id["non-mapping"]["created_at"] is None
        assert by_id["non-mapping"]["metadata"] == {}

    @pytest.mark.parametrize(
        ("session_id", "prefix", "expected_message"),
        [
            (
                "../session-traversal-sentinel-727",
                "",
                "Session ID must be a single filename component",
            ),
            (
                "safe-id",
                "../prefix-traversal-sentinel-727/",
                "Session prefix must be a single filename component",
            ),
        ],
    )
    def test_create_and_from_id_reject_unconfined_components(
        self,
        tmp_path,
        caplog,
        session_id,
        prefix,
        expected_message,
    ):
        """Creation and loading reject traversal before filesystem access."""
        base_dir = tmp_path / "sessions"
        base_dir.mkdir()

        for operation in (SessionManager.create, SessionManager.from_id):
            caplog.clear()
            with (
                caplog.at_level("INFO"),
                pytest.raises(
                    ValueError,
                    match=f"^{expected_message}$",
                ) as exc_info,
            ):
                operation(
                    base_dir=base_dir,
                    session_id=session_id,
                    prefix=prefix,
                )
            observable = f"{exc_info.value!r}\n{caplog.text}"
            assert "traversal-sentinel-727" not in observable

        assert list(base_dir.iterdir()) == []

    def test_create_and_from_id_reject_external_session_symlink(self, tmp_path):
        """A valid alias name cannot redirect a session outside its base."""
        base_dir = tmp_path / "sessions"
        base_dir.mkdir()
        external_session = tmp_path / "external"
        external_session.mkdir()
        alias = base_dir / ".alias"
        alias.symlink_to(external_session, target_is_directory=True)

        for operation in (SessionManager.create, SessionManager.from_id):
            with pytest.raises(ValueError, match="^Invalid session directory$"):
                operation(base_dir=base_dir, session_id="alias")

        assert alias.is_symlink()
        assert list(external_session.iterdir()) == []

    def test_from_id_does_not_follow_external_metadata_symlink(self, tmp_path):
        """Loading a confined session never follows its metadata outside the base."""
        secret = "from-id-metadata-symlink-sentinel-727"
        base_dir = tmp_path / "sessions"
        session_dir = base_dir / ".safe"
        session_dir.mkdir(parents=True)
        external_metadata = tmp_path / "external-metadata.json"
        external_metadata.write_text(
            json.dumps({"project_name": secret}),
            encoding="utf-8",
        )
        (session_dir / ".metadata.json").symlink_to(external_metadata)

        loaded = SessionManager.from_id("safe", base_dir)

        assert loaded.project_name == "unknown_project"
        assert loaded.metadata == {
            "session_id": "safe",
            "project_name": "unknown_project",
        }
        assert secret not in repr(loaded.metadata)
        assert external_metadata.read_text(encoding="utf-8") == json.dumps(
            {"project_name": secret}
        )

    def test_session_child_paths_reject_absolute_and_traversal(self, tmp_path):
        """Session path helpers reject portable escape syntax value-free."""
        session = SessionManager.create(tmp_path, session_id="confined")
        unsafe_calls = (
            lambda: session.get_subdir("../subdir-traversal-sentinel-727"),
            lambda: session.get_file(str(tmp_path / "absolute-sentinel-727")),
            lambda: session.get_file(r"C:\windows-absolute-sentinel-727"),
            lambda: session.get_file(r"\windows-rooted-sentinel-727"),
        )

        for unsafe_call in unsafe_calls:
            with pytest.raises(ValueError, match="^Invalid session path$") as exc_info:
                unsafe_call()
            assert "sentinel-727" not in str(exc_info.value)

        assert not (tmp_path / "subdir-traversal-sentinel-727").exists()

    def test_session_child_paths_confine_symlink_targets(self, tmp_path):
        """Internal aliases work, while aliases outside the session are rejected."""
        session = SessionManager.create(tmp_path, session_id="symlink-paths")
        internal = session.session_dir / "internal"
        internal.mkdir()
        (session.session_dir / "internal-alias").symlink_to(
            internal,
            target_is_directory=True,
        )

        nested = session.get_subdir("internal-alias/nested")

        assert nested == internal / "nested"
        assert nested.is_dir()
        assert session.get_file("internal-alias/result.json") == (
            internal / "result.json"
        )

        external = tmp_path / "external"
        external.mkdir()
        (session.session_dir / "external-alias").symlink_to(
            external,
            target_is_directory=True,
        )
        with pytest.raises(ValueError, match="^Invalid session path$"):
            session.get_subdir("external-alias/nested")
        with pytest.raises(ValueError, match="^Invalid session path$"):
            session.get_file("external-alias/result.json")

    def test_list_sessions_ignores_non_session_dirs(self, tmp_path):
        """Test that list_sessions ignores non-session directories."""
        # Create a session
        session = SessionManager.create(base_dir=tmp_path, session_id="real-session")
        session.save_metadata()

        # Create some non-session directories
        (tmp_path / "regular_dir").mkdir()
        (tmp_path / "another_dir").mkdir()

        # List sessions - should only find the one with prefix
        sessions = SessionManager.list_sessions(tmp_path)

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "real-session"

    def test_session_with_custom_prefix(self, tmp_path):
        """Test creating session with custom prefix."""
        session = SessionManager.create(
            base_dir=tmp_path, project_name="test", prefix="session_"
        )

        assert session.session_dir.name.startswith("session_")
        assert session.session_dir.exists()

    def test_repr_and_str(self, tmp_path):
        """Test string representations."""
        session = SessionManager.create(
            base_dir=tmp_path, project_name="test_project", session_id="test-123"
        )

        # Test repr
        repr_str = repr(session)
        assert "SessionManager" in repr_str
        assert "test-123" in repr_str

        # Test str
        str_str = str(session)
        assert "test-123" in str_str
        assert "test_project" in str_str
