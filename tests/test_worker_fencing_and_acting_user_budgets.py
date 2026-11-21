"""Regression tests for mid-processing deletion fencing, duration/reservation budgets, and acting-user billing."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.app import app
from src.database import db
from src.models import InternalShare, Recording, User
from src.services.recording_merge import MergeError, run_merge_job
from src.services.token_tracking import TokenTracker
from src.services.transcription_tracking import TranscriptionTracker
from src.tasks import processing as proc_mod


def _mk_user(prefix, token_budget=None, transcription_budget=None):
    uid = uuid.uuid4().hex[:8]
    u = User(
        username=f"{prefix}_{uid}",
        email=f"{prefix}_{uid}@example.com",
        password="x",
        email_verified=True,
        monthly_token_budget=token_budget,
        monthly_transcription_budget=transcription_budget,
    )
    db.session.add(u)
    db.session.commit()
    return u


def test_transcription_and_token_trackers_reserve_and_prevent_overrun():
    with app.app_context():
        user = _mk_user("resv", token_budget=1000, transcription_budget=600)
        uid = user.id

        tt = TranscriptionTracker()
        # Single job larger than remaining budget is denied immediately
        ok_big, _, _ = tt.check_budget(uid, additional_seconds=900)
        assert ok_big is False

        # Concurrent reservations cannot exceed cap
        ok1, _, _ = tt.reserve_budget(uid, additional_seconds=400)
        assert ok1 is True
        ok2, _, _ = tt.reserve_budget(uid, additional_seconds=300)
        assert ok2 is False

        # Releasing the first reservation allows the second to proceed
        tt.release_reservation(uid, 400)
        ok3, _, _ = tt.reserve_budget(uid, additional_seconds=300)
        assert ok3 is True
        tt.release_reservation(uid, 300)

        tok = TokenTracker()
        ok_t1, _, _ = tok.reserve_budget(uid, estimated_tokens=700)
        assert ok_t1 is True
        ok_t2, _, _ = tok.reserve_budget(uid, estimated_tokens=400)
        assert ok_t2 is False
        tok.release_reservation(uid, 700)
        ok_t3, _, _ = tok.reserve_budget(uid, estimated_tokens=400)
        assert ok_t3 is True
        tok.release_reservation(uid, 400)

        db.session.delete(db.session.get(User, uid))
        db.session.commit()


def test_background_title_summary_and_events_bill_acting_user_not_owner():
    with app.app_context():
        owner = _mk_user("bill_own")
        editor = _mk_user("bill_ed")
        rec = Recording(
            user_id=owner.id,
            title="Untitled",
            original_filename="clip.mp3",
            transcription="We agreed to ship the release next Tuesday at 2pm.",
            status="COMPLETED",
        )
        db.session.add(rec)
        db.session.commit()
        owner_id, editor_id, rec_id = owner.id, editor.id, rec.id

        called_user_ids = []

        def _fake_llm(*args, **kwargs):
            called_user_ids.append((kwargs.get("operation_type"), kwargs.get("user_id")))
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content='{"events": []}' if kwargs.get("operation_type") == "event_extraction" else "Generated Title"))]
            return resp

        with patch.object(proc_mod, "call_llm_completion", side_effect=_fake_llm):
            proc_mod._generate_ai_title(rec, acting_user_id=editor_id)
            proc_mod.extract_events_from_transcript(rec_id, rec.transcription, "Summary", acting_user_id=editor_id)

        assert ("title_generation", editor_id) in called_user_ids
        assert ("event_extraction", editor_id) in called_user_ids
        assert all(uid != owner_id for _, uid in called_user_ids)

        # Also test shared editor calling /recording/<id>/regenerate_title bills editor_id
        db.session.add(
            InternalShare(
                recording_id=rec_id,
                owner_id=owner_id,
                shared_with_user_id=editor_id,
                can_edit=True,
            )
        )
        db.session.commit()

    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(editor_id)
            s["_fresh"] = True

        seen_route = []
        def _fake_title_helper(r, acting_user_id=None):
            seen_route.append(acting_user_id)
            return "Updated Title"

        with patch("src.app.ENABLE_INTERNAL_SHARING", True),              patch("src.api.recordings.client", MagicMock()),              patch.object(proc_mod, "_generate_ai_title", side_effect=_fake_title_helper):
            resp = client.post(f"/recording/{rec_id}/regenerate_title")
            assert resp.status_code == 200
        assert seen_route == [editor_id]
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf
        with app.app_context():
            InternalShare.query.filter_by(recording_id=rec_id).delete()
            db.session.delete(db.session.get(Recording, rec_id))
            db.session.delete(db.session.get(User, editor_id))
            db.session.delete(db.session.get(User, owner_id))
            db.session.commit()


def test_merge_job_deletes_uploaded_object_when_recording_deleted_mid_upload(tmp_path):
    staging_file = tmp_path / "merged.m4a"
    staging_file.write_bytes(b"fake-m4a-data")

    with app.app_context():
        user = _mk_user("merge_fence")
        src1 = Recording(user_id=user.id, title="S1", original_filename="s1.mp3", audio_path="local:s1.mp3", status="COMPLETED")
        src2 = Recording(user_id=user.id, title="S2", original_filename="s2.mp3", audio_path="local:s2.mp3", status="COMPLETED")
        target = Recording(user_id=user.id, title="Merged", original_filename="merged.m4a", status="MERGING")
        db.session.add_all([src1, src2, target])
        db.session.commit()
        uid, s1_id, s2_id, tgt_id = user.id, src1.id, src2.id, target.id

        deleted_locators = []
        fake_storage = MagicMock()
        fake_storage.build_recording_key.return_value = "recordings/merged.m4a"

        def _fake_upload(local_path, key, **kwargs):
            # Simulate concurrent user deletion of target recording right as upload finishes
            db.session.execute(db.delete(Recording).where(Recording.id == tgt_id))
            db.session.commit()
            return MagicMock(locator="s3:recordings/merged.m4a")

        fake_storage.upload_local_file.side_effect = _fake_upload
        fake_storage.delete.side_effect = lambda loc, **kw: deleted_locators.append(loc)

        class _FakeMat:
            def __init__(self, p):
                self.local_path = str(p)
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        f1 = tmp_path / "s1.mp3"
        f2 = tmp_path / "s2.mp3"
        f1.write_bytes(b"111")
        f2.write_bytes(b"222")
        fake_storage.materialize.side_effect = lambda p: _FakeMat(f1 if "s1" in p else f2)

        def _fake_concat(paths, out_path):
            with open(out_path, "wb") as fp:
                fp.write(b"merged-bytes")

        with patch("src.services.recording_merge.get_storage_service", return_value=fake_storage),              patch("src.services.recording_merge._concat_to_m4a", side_effect=_fake_concat):
            with pytest.raises(MergeError):
                run_merge_job(tgt_id, [s1_id, s2_id], delete_originals=False)

        assert "s3:recordings/merged.m4a" in deleted_locators

        # Cleanup remaining rows
        for rid in (s1_id, s2_id):
            r = db.session.get(Recording, rid)
            if r:
                db.session.delete(r)
        u = db.session.get(User, uid)
        if u:
            db.session.delete(u)
        db.session.commit()
