# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the video_frames extraction helper."""

import hashlib
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from world_understanding.functions.cv.video_frames import (
    MAX_FRAMES,
    _pick_timestamps,
    _seek_and_save,
    extract_frames,
)


def _mp4v_encoder_available() -> bool:
    """Probe whether the bundled cv2 wheel can write mp4v on this platform."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = Path(f.name)
    try:
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
        ok = writer.isOpened()
        writer.release()
        return ok
    finally:
        path.unlink(missing_ok=True)


_MP4V_OK = _mp4v_encoder_available()
needs_mp4v = pytest.mark.skipif(
    not _MP4V_OK,
    reason="cv2.VideoWriter mp4v encoder unavailable on this platform",
)


def _write_video(path: Path, *, n_frames: int, fps: float = 30.0) -> Path:
    """Write a tiny varying-brightness video so VideoCapture can decode it."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height = 32, 32
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    assert writer.isOpened(), "encoder probe passed but writer.isOpened()=False"
    try:
        for i in range(n_frames):
            value = (i * 17) % 256
            frame = np.full((height, width, 3), value, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@needs_mp4v
class TestExtractFrames:
    """Tests for extract_frames."""

    @pytest.fixture
    def short_video(self, tmp_path: Path) -> Path:
        # 30 frames @ 30fps = 1.0 second.
        return _write_video(tmp_path / "clip.mp4", n_frames=30, fps=30.0)

    def test_happy_path_default_n(self, short_video: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "frames"
        paths = extract_frames(short_video, out_dir, n=3)

        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.parent == out_dir
            assert p.suffix == ".png"

        names = [p.name for p in paths]
        assert names == sorted(names)
        timestamps = [int(p.stem.split("__t")[1]) for p in paths]
        assert timestamps == sorted(timestamps)

    def test_returned_frames_are_distinct(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        # Hash bytes — fixture varies brightness per frame, so PNGs must differ.
        # Catches the silent duplicate-frame regression class
        # (unchecked seek, n > frame_count, keyframe-snap).
        paths = extract_frames(short_video, tmp_path / "out", n=5)
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == len(paths)

    def test_one_frame_video_returns_one_frame(self, tmp_path: Path) -> None:
        # Regression for the round-10 short-video bug: midpoint sampling on
        # a 1-frame source seeks past the only frame's PTS, returns False,
        # and on some backends the cursor cannot recover via
        # CAP_PROP_POS_FRAMES=0. extract_frames must detect this short-clip
        # case and route through the sequential path so a perfectly valid
        # 1-frame mp4 returns 1 PNG instead of raising
        # "No frames could be decoded".
        clip = _write_video(tmp_path / "one.mp4", n_frames=1, fps=30.0)
        paths = extract_frames(clip, tmp_path / "out", n=8)
        assert len(paths) == 1
        assert paths[0].exists()
        # Sequential path on a 1-frame source must land at index 0 with
        # the source's reported PTS (0ms for the only decoded frame).
        assert paths[0].name == "frame_000__t0.png"

    def test_n_capped_when_exceeds_frame_count(self, tmp_path: Path) -> None:
        # 4-frame source with n=8 in even mode: short-clip path must return
        # all 4 frames (never under-return because midpoint seeks pushed
        # the cursor past usable PTS).
        clip = _write_video(tmp_path / "tiny.mp4", n_frames=4, fps=30.0)
        paths = extract_frames(clip, tmp_path / "out", n=8)
        assert len(paths) == 4
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == 4, "short-clip path must return distinct frames"

    def test_stride_ms_caps_at_video_duration(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        # 1s clip with 500ms stride and n=8 should yield 2 frames (0ms, 500ms),
        # not 8. Also verify the timestamps embedded in the filenames are
        # monotonically increasing and land near the requested boundaries
        # within one source-frame interval.
        paths = extract_frames(short_video, tmp_path / "out", n=8, stride_ms=500)
        assert len(paths) == 2
        timestamps = [int(p.stem.split("__t")[1]) for p in paths]
        assert timestamps == sorted(timestamps)
        assert timestamps[0] < 50, (
            f"first sample should be near 0ms, got {timestamps[0]}"
        )
        # Second sample lands at the 500ms boundary or the next available
        # source frame (≈ 33ms later). Allow ±50ms slack for backend skew.
        assert 450 <= timestamps[1] <= 600, (
            f"second sample should be near 500ms, got {timestamps[1]}"
        )

    def test_too_short_video_break_early(self, tmp_path: Path) -> None:
        # 2-frame source, request 4: short-clip path returns both frames.
        clip = _write_video(tmp_path / "tiny.mp4", n_frames=2, fps=30.0)
        paths = extract_frames(clip, tmp_path / "out", n=4)
        assert len(paths) == 2
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == 2, "short-clip path must return distinct frames"

    def test_invalid_n_low(self, short_video: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"n must be in \[1, 64\]"):
            extract_frames(short_video, tmp_path / "out", n=0)

    def test_invalid_n_high(self, short_video: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"n must be in \[1, 64\]"):
            extract_frames(short_video, tmp_path / "out", n=MAX_FRAMES + 1)

    def test_invalid_stride(self, short_video: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"stride_ms must be positive"):
            extract_frames(short_video, tmp_path / "out", n=4, stride_ms=0)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_frames(tmp_path / "nope.mp4", tmp_path / "out")

    def test_unopenable_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a real video")
        with pytest.raises(RuntimeError, match=r"Could not open video"):
            extract_frames(bad, tmp_path / "out", n=2)

    def test_unopenable_file_cleans_fresh_output_dir(self, tmp_path: Path) -> None:
        # A failure on a freshly-created output_dir must clean up after itself.
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a real video")
        out = tmp_path / "frames-fresh"
        with pytest.raises(RuntimeError):
            extract_frames(bad, out, n=2)
        assert not out.exists()

    def test_small_stride_does_not_emit_duplicates(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        # 30fps source, stride_ms=1: every requested sample lands inside
        # the same source frame, so content-hash dedup must collapse them
        # to exactly one PNG. The function must NOT silently reroute to
        # the synth-clock fallback and start emitting fabricated stride
        # boundaries the user didn't ask for.
        paths = extract_frames(short_video, tmp_path / "out", n=8, stride_ms=1)
        assert len(paths) == 1

    def test_imwrite_failure_cleans_partial_pngs(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force cv2.imwrite to fail on the third call. The first two PNGs
        # land on disk; the function must raise and remove them as part of
        # the failure cleanup so the caller doesn't see partial output.
        out_dir = tmp_path / "frames"
        real_imwrite = cv2.imwrite
        call_count = {"n": 0}

        def flaky_imwrite(path: str, frame: object) -> bool:
            call_count["n"] += 1
            if call_count["n"] >= 3:
                return False
            return bool(real_imwrite(path, frame))

        monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)

        with pytest.raises(RuntimeError, match=r"Failed to write frame"):
            extract_frames(short_video, out_dir, n=5)
        # Output dir was created by extract_frames and is empty after cleanup.
        assert not any(out_dir.glob("*.png")), (
            "partial PNGs from before the imwrite failure were not cleaned up"
        )

    def test_metadata_fallback_runs_sequential_path(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When CAP_PROP_FRAME_COUNT/CAP_PROP_FPS report unusable values
        # (FFmpeg's `nb_frames`-missing case maps to -1 here), the helper
        # must take the sequential fallback and still return decodable
        # frames rather than raising.
        real_get = cv2.VideoCapture.get

        def stubborn_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1.0
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stubborn_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert 1 <= len(paths) <= 4
        for p in paths:
            assert p.exists()

    def test_partial_seek_success_falls_back_to_sequential(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression for the round-12 codex P2: when the seek path
        # partially succeeds (e.g. ``set(POS_MSEC, 0)`` is accepted as
        # a no-op but every later non-zero seek is rejected), the helper
        # used to write only the t=0 frame and skip the sequential
        # fallback because ``written`` was non-empty. With round-13's
        # ``seek_completed`` signal, that case should discard the partial
        # output and retry sequentially so the caller still gets at
        # least the requested number of stride samples it can compute.
        real_set = cv2.VideoCapture.set

        def picky_set(self: cv2.VideoCapture, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                # Accept seek to 0 as a no-op; reject all other seeks.
                return value == 0.0
            return bool(real_set(self, prop, value))

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        # 1s clip, stride 500ms, n=8: seek schedule is [0, 500].
        # Without the round-13 fix, only the t=0 frame would be written
        # and the second seek to 500ms would silently break the loop.
        # With the fix, the partial output is discarded and sequential
        # read fills in the schedule.
        paths = extract_frames(short_video, tmp_path / "out", n=8, stride_ms=500)
        assert len(paths) >= 2, (
            f"partial seek success must trigger sequential retry; got {len(paths)}"
        )

    def test_partial_seek_cleanup_ignores_unlink_errors(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_set = cv2.VideoCapture.set
        real_unlink = Path.unlink

        def picky_set(self: cv2.VideoCapture, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return value == 0.0
            return bool(real_set(self, prop, value))

        def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
            if self.suffix == ".png":
                raise OSError("temporary unlink failure")
            return real_unlink(self, *args, **kwargs)

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        paths = extract_frames(short_video, tmp_path / "out", n=8, stride_ms=500)

        assert len(paths) >= 2

    def test_reopen_after_zero_frame_seek_path(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression for the round-11 reopen-on-zero-frames path: when the
        # seek path produces zero PNGs against a metadata-good clip, the
        # helper must release+reopen the VideoCapture (CAP_PROP_POS_FRAMES=0
        # is unreliable after seek-past-EOF on some backends) and retry
        # via _read_sequential.
        # We force the seek path to return zero by failing every
        # CAP_PROP_POS_MSEC set() AND track that VideoCapture is
        # constructed twice (once initially, once on reopen).
        real_set = cv2.VideoCapture.set
        real_init = cv2.VideoCapture.__init__
        construct_count = {"n": 0}

        def picky_set(self: cv2.VideoCapture, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return False
            return bool(real_set(self, prop, value))

        def counted_init(self: cv2.VideoCapture, *args, **kwargs):  # type: ignore[no-untyped-def]
            construct_count["n"] += 1
            return real_init(self, *args, **kwargs)

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        monkeypatch.setattr(cv2.VideoCapture, "__init__", counted_init)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 1
        # The reopen branch must have been taken: the original open plus
        # the defensive close-and-reopen for the sequential fallback.
        assert construct_count["n"] >= 2, (
            f"reopen path was not exercised; constructed {construct_count['n']} time(s)"
        )

    def test_seek_unsupported_falls_back_to_sequential(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some FFmpeg builds report frame_count/fps correctly but reject
        # every CAP_PROP_POS_MSEC seek with set()->False. The helper must
        # fall back to sequential reads instead of raising
        # "No frames could be decoded from".
        real_set = cv2.VideoCapture.set

        def picky_set(self: cv2.VideoCapture, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return False
            return bool(real_set(self, prop, value))

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 1
        for p in paths:
            assert p.exists()

    def test_pos_frames_advances_while_pos_msec_stuck(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin CAP_PROP_POS_MSEC at 0 while letting CAP_PROP_POS_FRAMES
        # advance (round-3 contract). Dedup must accept on the
        # POS_FRAMES signal alone, so we still get multiple distinct
        # PNGs from a clearly-decodable stream.
        real_get = cv2.VideoCapture.get

        def stuck_msec_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stuck_msec_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 2
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == len(paths)

    def test_seek_path_filenames_have_no_index_gaps(
        self, short_video: Path, tmp_path: Path
    ) -> None:
        # extract_frames returns chronologically-named files with
        # contiguous indices, even when dedup skips iterations.
        # Use stride_ms=33 so we get multiple distinct frames AND
        # each iteration has a chance to dedup-skip if the backend
        # behavior changes — covers the contract end-to-end rather
        # than passing trivially on a single-frame return.
        paths = extract_frames(short_video, tmp_path / "out", n=8, stride_ms=33)
        assert len(paths) >= 2, "test setup must yield multiple frames"
        names = [p.name for p in paths]
        for i, name in enumerate(names):
            assert name.startswith(f"frame_{i:03d}__t")

    def test_existing_output_dir_preserved_on_failure(self, tmp_path: Path) -> None:
        # If output_dir already existed before the call, failure must
        # leave the dir AND any unrelated files inside it untouched.
        out = tmp_path / "preexisting"
        out.mkdir()
        marker = out / "user_data.txt"
        marker.write_text("do not delete")

        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a real video")
        with pytest.raises(RuntimeError, match=r"Could not open video"):
            extract_frames(bad, out, n=2)

        assert out.exists() and out.is_dir()
        assert marker.exists()
        assert marker.read_text() == "do not delete"

    def test_failure_cleanup_ignores_unlink_errors(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real_imwrite = cv2.imwrite

        def flaky_imwrite(path: str, frame: object) -> bool:
            if Path(path).name.startswith("frame_001"):
                return False
            return bool(real_imwrite(path, frame))

        monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)
        monkeypatch.setattr(
            Path, "unlink", lambda self: (_ for _ in ()).throw(OSError("busy"))
        )

        with pytest.raises(RuntimeError, match=r"Failed to write frame"):
            extract_frames(short_video, tmp_path / "out", n=3)

    def test_fresh_output_dir_cleanup_ignores_rmdir_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"not a real video")
        out = tmp_path / "frames-rmdir-fails"

        monkeypatch.setattr(
            Path, "rmdir", lambda self: (_ for _ in ()).throw(OSError("busy"))
        )

        with pytest.raises(RuntimeError, match=r"Could not open video"):
            extract_frames(bad, out, n=2)

        assert out.exists()

    def test_seek_path_falls_back_when_counters_stuck(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Backend reports valid metadata so the seek path runs, but
        # both POS_FRAMES and POS_MSEC stay pinned at 0 after every
        # successful read. Content-hash dedup is robust to that: the
        # frames are visually distinct even when the position counters
        # lie, so extract_frames produces multiple PNGs without
        # needing to fall through to _read_sequential.
        real_get = cv2.VideoCapture.get

        def stuck_pos_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop in (cv2.CAP_PROP_POS_FRAMES, cv2.CAP_PROP_POS_MSEC):
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stuck_pos_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 2
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == len(paths)

    def test_sequential_fallback_bounds_work_on_static_long_clip(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A long static-scene video with bad metadata used to drive
        # _read_sequential to decode every frame to EOF (each one
        # dedup-skipped) just to write 1 PNG. The dedup-skip budget
        # must stop the loop after a bounded number of duplicate
        # decodes per requested output. We force the fallback path
        # by stubbing FRAME_COUNT/FPS, write a 600-frame static clip
        # (20s at 30fps), and assert that read() is called far fewer
        # times than the full clip length.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = tmp_path / "long_static.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
        if not writer.isOpened():
            pytest.skip("mp4v encoder unavailable")
        try:
            for _ in range(600):
                writer.write(np.full((32, 32, 3), 200, dtype=np.uint8))
        finally:
            writer.release()

        real_get = cv2.VideoCapture.get

        def junk_meta_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS):
                return -1.0
            return float(real_get(self, prop))

        real_read = cv2.VideoCapture.read
        read_calls = {"n": 0}

        def counted_read(self: cv2.VideoCapture):  # type: ignore[no-untyped-def]
            read_calls["n"] += 1
            return real_read(self)

        monkeypatch.setattr(cv2.VideoCapture, "get", junk_meta_get)
        monkeypatch.setattr(cv2.VideoCapture, "read", counted_read)
        paths = extract_frames(path, tmp_path / "out", n=4)
        assert len(paths) == 1, "static scene should dedup to 1 PNG"
        # Budget: max(60, n*30) = 120 dedup-skips + ~1 accepted read,
        # plus any dedup-skips before the first accept (none here, since
        # frame 0 is always accepted). Generous upper bound here that
        # would still fail the "decode-until-EOF" regression (which
        # would land at 600).
        assert read_calls["n"] <= 200, (
            f"sequential fallback decoded {read_calls['n']} frames; "
            f"dedup-skip budget should have stopped it well below 200"
        )

    def test_static_scene_dedup_path_symmetric(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Same static-scene source must produce the same dedup'd
        # output whether extract_frames goes through the seek path
        # (good metadata) or the sequential fallback (forced bad
        # metadata). Without sequential-path dedup, the bad-metadata
        # case would emit n byte-identical PNGs.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = tmp_path / "static.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (32, 32))
        if not writer.isOpened():
            pytest.skip("mp4v encoder unavailable")
        try:
            for _ in range(30):
                # Constant gray frames — identical bytes per decode.
                writer.write(np.full((32, 32, 3), 128, dtype=np.uint8))
        finally:
            writer.release()

        seek_paths = extract_frames(path, tmp_path / "seek_out", n=8)
        assert len(seek_paths) == 1, "seek path should dedup static scene to 1 PNG"

        real_get = cv2.VideoCapture.get

        def junk_meta_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop in (cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FPS):
                return -1.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", junk_meta_get)
        seq_paths = extract_frames(path, tmp_path / "seq_out", n=8)
        assert len(seq_paths) == 1, (
            "sequential fallback should also dedup static scene to 1 PNG"
        )

    def test_pos_msec_stuck_does_not_collapse_filenames_to_t0(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Backend pins POS_MSEC at 0 for every seek. With content-hash
        # dedup, distinct decoded frames are still accepted — but the
        # filename t<ms> suffix must NOT collapse to __t0 for every
        # frame. The helper must detect the stuck reading and fall
        # back to the requested t_ms in the filename.
        real_get = cv2.VideoCapture.get

        def stuck_msec_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stuck_msec_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 2
        timestamps = [int(p.stem.split("__t")[1]) for p in paths]
        # Distinct frames must carry distinct timestamp suffixes.
        assert len(set(timestamps)) == len(timestamps)

    def test_metadata_fallback_uses_valid_fps_when_only_frame_count_bad(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # FRAME_COUNT is junk but FPS is valid (60 here). The metadata
        # fallback must pass that fps through to _read_sequential so
        # stride sampling honors the real frame rate. With the old
        # code (fps_hint=None always) the synth clock would fall back
        # to 30fps and mis-space stride samples by 2x.
        clip = _write_video(tmp_path / "60fps.mp4", n_frames=120, fps=60.0)
        real_get = cv2.VideoCapture.get

        def junk_frame_count_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1.0
            if prop == cv2.CAP_PROP_POS_MSEC:
                # Force the synth clock — otherwise raw POS_MSEC drives
                # the stride decision and fps_hint never matters.
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", junk_frame_count_get)
        paths = extract_frames(clip, tmp_path / "out", n=4, stride_ms=1000)
        # 120 frames at 60fps = 2s. With proper 60fps synth clock,
        # stride 1000ms accepts at boundaries 0 and 1000 → exactly 2.
        # With the broken 30fps default we'd have run out of frames
        # earlier or mis-spaced.
        assert len(paths) == 2

    def test_sequential_stride_anchors_to_requested_schedule(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When a decoded frame overshoots a requested stride boundary,
        # the next target must still be the next k * stride_ms boundary
        # past the accepted frame — NOT (accepted_t + stride_ms),
        # which would shift every later target by the overshoot and
        # silently skip valid scheduled samples. Tested with a 2fps
        # source whose decode timestamps land on 0/500/1000/... and
        # stride_ms=750 (a 750ms target sits between two decode
        # boundaries).
        clip = _write_video(tmp_path / "2fps.mp4", n_frames=8, fps=2.0)
        real_set = cv2.VideoCapture.set

        def picky_set(self: cv2.VideoCapture, prop: int, val: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return False
            return bool(real_set(self, prop, val))

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        paths = extract_frames(clip, tmp_path / "out", n=6, stride_ms=750)
        timestamps = [int(p.stem.split("__t")[1]) for p in paths]
        # The accepted frames should land on or just past every
        # multiple-of-750ms boundary the source can resolve. With the
        # buggy `next_target = t_ms + stride_ms` advancement, the
        # 1500ms boundary would be skipped because the 1000ms accept
        # would set next=1750.
        assert any(1500 <= t < 2000 for t in timestamps)

    def test_seek_unsupported_fallback_uses_reported_fps(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # 60fps source where the seek path is rejected — sequential
        # fallback must use the *reported* fps for its synth clock.
        # With the old hard-coded 30fps default, stride_ms=1000 on a
        # 60fps clip would mis-space samples (every 30 decoded frames
        # ≈ every 500ms of source) instead of every 1000ms.
        clip = _write_video(tmp_path / "60fps.mp4", n_frames=120, fps=60.0)
        real_set = cv2.VideoCapture.set

        def picky_set(self: cv2.VideoCapture, prop: int, value: float) -> bool:
            if prop == cv2.CAP_PROP_POS_MSEC:
                return False
            return bool(real_set(self, prop, value))

        monkeypatch.setattr(cv2.VideoCapture, "set", picky_set)
        paths = extract_frames(clip, tmp_path / "out", n=4, stride_ms=1000)
        # 120 frames at 60fps = 2.0s. With proper fps-hint synth clock
        # (16.67ms/frame), accepting at 1000ms boundaries gives exactly
        # 2 frames (t=0 and t=1000). With the old 30fps default
        # (33.33ms/frame), the loop would reach n=4 within 120 frames.
        assert len(paths) == 2
        timestamps = [int(p.stem.split("__t")[1]) for p in paths]
        # Both samples must land on stride boundaries (mod 1000 == 0).
        assert all(t % 1000 == 0 for t in timestamps)

    def test_metadata_fallback_synth_clock_drives_stride(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Pin both metadata AND POS_MSEC: forces _read_sequential to
        # synthesize a 30fps frame-index clock for stride filtering.
        # Round-1 contract — without the synth clock, stride mode
        # against a stuck POS_MSEC backend would lock to one PNG.
        real_get = cv2.VideoCapture.get

        def stuck_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1.0
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            if prop == cv2.CAP_PROP_POS_MSEC:
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stuck_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4, stride_ms=33)
        assert len(paths) >= 2
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == len(paths)

    def test_sequential_fallback_imwrite_failure_cleanup(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mirror of test_imwrite_failure_cleans_partial_pngs but for the
        # _read_sequential branch (R5 added the symmetric out.unlink on
        # imwrite failure there). Forces metadata_ok=False so the
        # sequential path runs, then fails imwrite mid-loop.
        real_get = cv2.VideoCapture.get
        real_imwrite = cv2.imwrite

        def stub_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return -1.0
            if prop == cv2.CAP_PROP_FPS:
                return 0.0
            return float(real_get(self, prop))

        call_count = {"n": 0}

        def flaky_imwrite(path: str, frame: object) -> bool:
            call_count["n"] += 1
            if call_count["n"] >= 3:
                return False
            return bool(real_imwrite(path, frame))

        monkeypatch.setattr(cv2.VideoCapture, "get", stub_get)
        monkeypatch.setattr(cv2, "imwrite", flaky_imwrite)
        out_dir = tmp_path / "frames-seq"
        with pytest.raises(RuntimeError, match=r"Failed to write frame"):
            extract_frames(short_video, out_dir, n=5)
        assert not any(out_dir.glob("*.png"))

    def test_pos_msec_advances_while_pos_frames_stuck(
        self,
        short_video: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inverse of test_pos_frames_advances_while_pos_msec_stuck:
        # backend pins POS_FRAMES at 0 but POS_MSEC advances normally.
        # Round-5 dedup must accept on the ms signal alone — relying
        # solely on POS_FRAMES would lock to one PNG.
        real_get = cv2.VideoCapture.get

        def stuck_frames_get(self: cv2.VideoCapture, prop: int) -> float:
            if prop == cv2.CAP_PROP_POS_FRAMES:
                return 0.0
            return float(real_get(self, prop))

        monkeypatch.setattr(cv2.VideoCapture, "get", stuck_frames_get)
        paths = extract_frames(short_video, tmp_path / "out", n=4)
        assert len(paths) >= 2
        digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        assert len(digests) == len(paths)


class TestPickTimestamps:
    """Direct tests for the pure timestamp picker (no cv2 dependency)."""

    def test_even_n8_duration_1000ms(self) -> None:
        ts = _pick_timestamps(duration_ms=1000.0, n=8, stride_ms=None)
        assert ts == [62.5, 187.5, 312.5, 437.5, 562.5, 687.5, 812.5, 937.5]

    def test_even_n1_returns_midpoint(self) -> None:
        assert _pick_timestamps(duration_ms=500.0, n=1, stride_ms=None) == [250.0]

    def test_stride_n1_returns_zero(self) -> None:
        # n=1 with stride starts at t=0, not at the midpoint — documented.
        assert _pick_timestamps(duration_ms=500.0, n=1, stride_ms=100) == [0.0]

    def test_stride_drops_past_duration(self) -> None:
        ts = _pick_timestamps(duration_ms=350.0, n=8, stride_ms=100)
        assert ts == [0.0, 100.0, 200.0, 300.0]

    def test_even_non_positive_n_returns_empty(self) -> None:
        assert _pick_timestamps(duration_ms=350.0, n=0, stride_ms=None) == []

    def test_even_strictly_interior(self) -> None:
        # No timestamp at or beyond duration_ms; no timestamp at exactly 0.
        ts = _pick_timestamps(duration_ms=10.0, n=64, stride_ms=None)
        assert all(0.0 < t < 10.0 for t in ts)
        assert ts == sorted(ts)


def test_seek_and_save_returns_false_when_read_after_seek_fails(tmp_path: Path) -> None:
    class FakeCapture:
        def set(self, _prop: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

    paths: list[Path] = []

    assert _seek_and_save(FakeCapture(), [0.0], tmp_path, paths) is False
    assert paths == []
