"""ASR_MIN_SPEAKERS / ASR_MAX_SPEAKERS with ASR_BASE_URL alone (#415).

The ASR connector is chosen from ASR_BASE_URL, and the docs mark the old
USE_ASR_ENDPOINT flag as deprecated, but the speaker-count defaults were only
read when that flag was set. An installation set up as documented sent no
speaker counts to WhisperX, also for the watch folder.

app_config reads the environment at import, so each case imports it in a
fresh interpreter.
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = (
    "import json, src.config.app_config as c; "
    "print(json.dumps([c.ASR_MIN_SPEAKERS, c.ASR_MAX_SPEAKERS]))"
)


def _read(**env):
    clean = {k: v for k, v in os.environ.items()
             if k not in ("USE_ASR_ENDPOINT", "ASR_BASE_URL", "TRANSCRIPTION_CONNECTOR",
                          "ASR_MIN_SPEAKERS", "ASR_MAX_SPEAKERS")}
    clean.update(env)
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=ROOT, env=clean,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("env", [
    {"ASR_BASE_URL": "http://whisperx:9000"},                              # as documented
    {"TRANSCRIPTION_CONNECTOR": "asr_endpoint", "ASR_BASE_URL": "http://whisperx:9000"},
    {"USE_ASR_ENDPOINT": "true", "ASR_BASE_URL": "http://whisperx:9000"},  # legacy flag
])
def test_speaker_counts_are_read_whenever_the_asr_connector_is_active(env):
    assert _read(ASR_MIN_SPEAKERS="2", ASR_MAX_SPEAKERS="4", **env) == ["2", "4"]


def test_speaker_counts_are_ignored_without_an_asr_connector():
    assert _read(ASR_MIN_SPEAKERS="2", ASR_MAX_SPEAKERS="4") == [None, None]


def test_the_resolver_hands_the_env_defaults_to_every_path(monkeypatch):
    import src.config.app_config as c
    from src.app import app
    from src.services.transcription_defaults import resolve_transcription_params
    monkeypatch.setattr(c, "ASR_MIN_SPEAKERS", "2")
    monkeypatch.setattr(c, "ASR_MAX_SPEAKERS", "4")
    with app.app_context():
        resolved = resolve_transcription_params(None, {}, tags=[], folder=None, owner=None)
        assert (resolved["min_speakers"], resolved["max_speakers"]) == (2, 4)
        resolved = resolve_transcription_params(None, {"min_speakers": 3}, tags=[], folder=None, owner=None)
        assert (resolved["min_speakers"], resolved["max_speakers"]) == (3, 4)


def _tag(min_s=None, max_s=None):
    from types import SimpleNamespace
    return SimpleNamespace(id=1, default_language=None, default_min_speakers=min_s, default_max_speakers=max_s,
                           default_hotwords=None, default_initial_prompt=None, default_transcription_model=None)


@pytest.mark.parametrize("env_min,env_max,overrides,tag,expected", [
    ("2", "5", {}, None, (2, 5)),                          # environment alone
    (None, "5", {}, (6, None), (6, 6)),                    # tag minimum over the env maximum: the tag wins
    ("4", None, {"max_speakers": 2}, None, (2, 2)),        # request maximum under the env minimum: the request wins
    ("2", "5", {"min_speakers": 3, "max_speakers": 3}, None, (3, 3)),  # exact count from the form
    ("2", "5", {}, (1, 8), (1, 8)),                        # tag range replaces the env range
])
def test_speaker_bounds_from_several_levels_stay_valid(monkeypatch, env_min, env_max, overrides, tag, expected):
    import src.config.app_config as c
    from src.app import app
    from src.services.transcription_defaults import resolve_transcription_params
    monkeypatch.setattr(c, "ASR_MIN_SPEAKERS", env_min)
    monkeypatch.setattr(c, "ASR_MAX_SPEAKERS", env_max)
    tags = [_tag(*tag)] if tag else []
    with app.app_context():
        r = resolve_transcription_params(None, overrides, tags=tags, folder=None, owner=None)
    assert (r["min_speakers"], r["max_speakers"]) == expected
