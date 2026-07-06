#!/usr/bin/env python3
"""
Regression tests for the ffmpeg/ffprobe subprocess timeout hardening.

An ffmpeg invocation with no timeout can hang a job-queue worker forever on a
malformed/adversarial media file. These verify the shared executor now bounds
every call and translates a timeout into a clean FFmpegError instead of
blocking indefinitely.

Hermetic: subprocess is mocked, no real ffmpeg needed.
"""

import subprocess
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app
from src.utils import ffmpeg_utils
from src.utils.ffmpeg_utils import _run_ffmpeg_command, FFmpegError


def test_run_ffmpeg_passes_timeout():
    with app.app_context():
        with mock.patch("src.utils.ffmpeg_utils.subprocess.run") as m:
            m.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
            _run_ffmpeg_command(["ffmpeg", "-i", "x"], "test op")
            assert m.call_count == 1
            _, kwargs = m.call_args
            assert kwargs.get("timeout") == ffmpeg_utils.FFMPEG_TIMEOUT_SECONDS
            assert kwargs.get("timeout") and kwargs["timeout"] > 0


def test_run_ffmpeg_timeout_raises_ffmpeg_error():
    with app.app_context():
        with mock.patch("src.utils.ffmpeg_utils.subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)):
            try:
                _run_ffmpeg_command(["ffmpeg", "-i", "x"], "stalled op")
                assert False, "expected FFmpegError on timeout"
            except FFmpegError as e:
                assert "timed out" in str(e).lower()


def test_ffprobe_probe_defaults_to_a_bounded_timeout():
    """probe() with no explicit timeout must still bound the call so a caller
    that forgot to pass one can't hang forever."""
    from src.utils import ffprobe

    captured = {}

    class FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return b'{"streams": [], "format": {}}', b""

        def kill(self):
            pass

    with mock.patch("src.utils.ffprobe.subprocess.Popen", return_value=FakeProc()):
        ffprobe.probe("/tmp/x.mp3")  # no timeout arg
        assert captured["timeout"] is not None and captured["timeout"] > 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
