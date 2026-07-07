"""Tests for the shared transcription-param resolver.

resolve_transcription_params is the single source of truth every ingestion path
(upload, reprocess, merge, stitch, share, auto-process) uses to turn a recording
+ optional per-request overrides into the transcribe job params. These tests pin
the precedence chain: override > tag > folder > env > owner > admin default.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, db
from src.models import User, Recording, Tag, Folder, RecordingTag
from src.services.transcription_defaults import resolve_transcription_params


def _mk_user(**kw):
    u = User(username=f"td_{uuid.uuid4().hex[:8]}", email=f"td_{uuid.uuid4().hex[:8]}@local.test", password="x", **kw)
    db.session.add(u)
    db.session.commit()
    return u


def _mk_recording(user, **kw):
    rec = Recording(user_id=user.id, title="r", status="COMPLETED",
                    audio_path="local://x.mp3", original_filename="x.mp3", **kw)
    db.session.add(rec)
    db.session.commit()
    return rec


def test_owner_defaults_when_no_tag_or_folder():
    with app.app_context():
        user = _mk_user(transcription_language="fr", transcription_hotwords="alpha",
                        transcription_initial_prompt="beta")
        rec = _mk_recording(user)
        p = resolve_transcription_params(rec)
        assert p["language"] == "fr"
        assert p["hotwords"] == "alpha"
        assert p["initial_prompt"] == "beta"
        assert p["tag_id"] is None


def test_tag_defaults_override_owner():
    with app.app_context():
        user = _mk_user(transcription_language="fr", transcription_hotwords="owner_hw")
        tag = Tag(name=f"t_{uuid.uuid4().hex[:6]}", user_id=user.id,
                  default_language="de", default_hotwords="tag_hw",
                  default_min_speakers=2, default_max_speakers=5)
        db.session.add(tag)
        db.session.commit()
        rec = _mk_recording(user)
        db.session.add(RecordingTag(recording_id=rec.id, tag_id=tag.id, order=1))
        db.session.commit()
        p = resolve_transcription_params(rec)
        assert p["language"] == "de"          # tag beats owner
        assert p["hotwords"] == "tag_hw"
        assert p["min_speakers"] == 2
        assert p["max_speakers"] == 5
        assert p["tag_id"] == tag.id


def test_folder_fills_gaps_not_covered_by_tag():
    with app.app_context():
        user = _mk_user()
        folder = Folder(name=f"f_{uuid.uuid4().hex[:6]}", user_id=user.id,
                        default_hotwords="folder_hw", default_min_speakers=3)
        tag = Tag(name=f"t_{uuid.uuid4().hex[:6]}", user_id=user.id, default_language="es")
        db.session.add_all([folder, tag])
        db.session.commit()
        rec = _mk_recording(user, folder_id=folder.id)
        db.session.add(RecordingTag(recording_id=rec.id, tag_id=tag.id, order=1))
        db.session.commit()
        p = resolve_transcription_params(rec)
        assert p["language"] == "es"          # from tag
        assert p["hotwords"] == "folder_hw"   # tag had none -> folder fills
        assert p["min_speakers"] == 3         # folder fills


def test_explicit_override_beats_everything():
    with app.app_context():
        user = _mk_user(transcription_hotwords="owner_hw")
        tag = Tag(name=f"t_{uuid.uuid4().hex[:6]}", user_id=user.id, default_hotwords="tag_hw")
        db.session.add(tag)
        db.session.commit()
        rec = _mk_recording(user)
        db.session.add(RecordingTag(recording_id=rec.id, tag_id=tag.id, order=1))
        db.session.commit()
        p = resolve_transcription_params(rec, {"hotwords": "override_hw", "min_speakers": "4"})
        assert p["hotwords"] == "override_hw"
        assert p["min_speakers"] == 4          # coerced from string


def test_empty_language_falls_through_to_defaults():
    with app.app_context():
        user = _mk_user(transcription_language="fr")
        rec = _mk_recording(user)
        # An empty/blank language is NOT a terminal auto-detect; it falls through
        # the chain to the account default 'fr'.
        assert resolve_transcription_params(rec, {"language": ""})["language"] == "fr"
        assert resolve_transcription_params(rec, {"language": "   "})["language"] == "fr"
        # Absent language behaves the same.
        assert resolve_transcription_params(rec, {"hotwords": "x"})["language"] == "fr"


def test_auto_detect_only_when_no_default_anywhere():
    with app.app_context():
        user = _mk_user()  # no account default language
        rec = _mk_recording(user)
        # Nothing in the chain sets a language => auto-detect (None).
        assert resolve_transcription_params(rec, {"language": ""})["language"] is None


def test_resolver_accepts_explicit_tags_folder_owner_without_recording():
    # Upload resolves before the recording row exists.
    with app.app_context():
        user = _mk_user(transcription_hotwords="acct_hw")
        tag = Tag(name=f"t_{uuid.uuid4().hex[:6]}", user_id=user.id, default_language="de")
        db.session.add(tag)
        db.session.commit()
        p = resolve_transcription_params(tags=[tag], folder=None, owner=user)
        assert p["language"] == "de"
        assert p["hotwords"] == "acct_hw"
        assert p["tag_id"] == tag.id


def test_explicit_tag_id_none_is_respected():
    with app.app_context():
        user = _mk_user()
        tag = Tag(name=f"t_{uuid.uuid4().hex[:6]}", user_id=user.id)
        db.session.add(tag)
        db.session.commit()
        rec = _mk_recording(user)
        db.session.add(RecordingTag(recording_id=rec.id, tag_id=tag.id, order=1))
        db.session.commit()
        # caller explicitly suppresses the tag prompt
        p = resolve_transcription_params(rec, {"tag_id": None})
        assert p["tag_id"] is None
        # absent -> first tag
        p2 = resolve_transcription_params(rec)
        assert p2["tag_id"] == tag.id
