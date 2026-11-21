"""Regression tests for API v1 input sanitization, tag permissions,
storage-backed deletion, template rendering safety, and LLM budget enforcement.
"""

import os
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.app as app_module
import src.services.llm as llm_module
from src.app import app, db
from src.models import Recording, User
from src.models.organization import Group, GroupMembership, RecordingTag, Tag
from src.models.processing_job import ProcessingJob
from src.models.sharing import InternalShare
from src.services.llm import TokenBudgetExceeded, call_chat_completion, call_llm_completion

app.config["WTF_CSRF_ENABLED"] = False

REPO_ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def _db():
    with app.app_context():
        yield


def _mk_user(prefix):
    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"{prefix}_{suffix}",
        email=f"{prefix}_{suffix}@local.test",
        password="x",
    )
    db.session.add(user)
    db.session.commit()
    return user.id


def _client_for(user_id):
    client = app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user_id)
        sess["_fresh"] = True
    return client


def _cleanup(user_ids=(), recording_ids=(), tag_ids=(), group_ids=()):
    with _db():
        for rid in recording_ids:
            RecordingTag.query.filter_by(recording_id=rid).delete()
            InternalShare.query.filter_by(recording_id=rid).delete()
            ProcessingJob.query.filter_by(recording_id=rid).delete()
            rec = db.session.get(Recording, rid)
            if rec:
                db.session.delete(rec)
        for tid in tag_ids:
            RecordingTag.query.filter_by(tag_id=tid).delete()
            tag = db.session.get(Tag, tid)
            if tag:
                db.session.delete(tag)
        for gid in group_ids:
            GroupMembership.query.filter_by(group_id=gid).delete()
            grp = db.session.get(Group, gid)
            if grp:
                db.session.delete(grp)
        for uid in user_ids:
            InternalShare.query.filter(
                (InternalShare.owner_id == uid) | (InternalShare.shared_with_user_id == uid)
            ).delete()
            user = db.session.get(User, uid)
            if user:
                db.session.delete(user)
        db.session.commit()


# ---------------------------------------------------------------------------
# Template v-html safety & API v1 summary/notes sanitization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", [
    "templates/components/detail/desktop-summary-tab.html",
    "templates/components/detail/mobile-summary-panel.html",
    "templates/components/detail/mobile-notes-panel.html",
    "templates/inquire.html",
])
def test_templates_never_pass_raw_fallback_to_v_html(rel_path):
    content = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert "v-html=" in content
    assert "|| selectedRecording.summary" not in content
    assert "|| selectedRecording.notes" not in content
    assert "|| message.content" not in content


def test_api_v1_sanitizes_summary_and_notes_across_write_routes():
    with _db():
        uid = _mk_user("san")
        rec = Recording(user_id=uid, title="t", status="COMPLETED")
        db.session.add(rec)
        db.session.commit()
        rid = rec.id

    payload = "<iframe src=x></iframe><details open ontoggle=alert(1)>x</details><b>ok</b>"
    try:
        client = _client_for(uid)

        r1 = client.patch(f"/api/v1/recordings/{rid}", json={"summary": payload, "notes": payload})
        assert r1.status_code == 200, r1.data
        with _db():
            rec = db.session.get(Recording, rid)
            assert "<iframe" not in (rec.summary or "").lower()
            assert "<details" not in (rec.summary or "").lower()
            assert "ontoggle" not in (rec.summary or "").lower()
            assert "<b>ok</b>" in (rec.summary or "")
            assert "<iframe" not in (rec.notes or "").lower()
            assert "<details" not in (rec.notes or "").lower()

        r2 = client.put(f"/api/v1/recordings/{rid}/notes", json={"notes": payload})
        assert r2.status_code == 200, r2.data
        assert "<iframe" not in r2.get_json()["notes"].lower()
        assert "<details" not in r2.get_json()["notes"].lower()

        r3 = client.put(f"/api/v1/recordings/{rid}/summary", json={"summary": payload})
        assert r3.status_code == 200, r3.data
        assert "<iframe" not in r3.get_json()["summary"].lower()
        assert "<details" not in r3.get_json()["summary"].lower()
    finally:
        _cleanup(user_ids=[uid], recording_ids=[rid])


# ---------------------------------------------------------------------------
# API v1 POST /recordings/<id>/tags permission enforcement
# ---------------------------------------------------------------------------

def test_api_v1_add_tags_blocks_view_only_share_recipient_and_non_admin_editor():
    with _db():
        owner_id = _mk_user("tag_own")
        viewer_id = _mk_user("tag_view")
        editor_id = _mk_user("tag_edit")
        rec = Recording(user_id=owner_id, title="owner rec", status="COMPLETED")
        db.session.add(rec)
        db.session.flush()
        rid = rec.id

        # View-only recipient owns a group where they are admin
        grp = Group(name=f"g_{uuid.uuid4().hex[:6]}")
        db.session.add(grp)
        db.session.flush()
        gid = grp.id
        db.session.add(GroupMembership(group_id=gid, user_id=viewer_id, role="admin"))
        db.session.add(GroupMembership(group_id=gid, user_id=editor_id, role="member"))
        db.session.add(GroupMembership(group_id=gid, user_id=owner_id, role="member"))

        gtag = Tag(name="grp-tag", user_id=viewer_id, group_id=gid, auto_share_on_apply=True)
        db.session.add(gtag)
        db.session.flush()
        tid = gtag.id

        db.session.add(InternalShare(
            recording_id=rid, owner_id=owner_id, shared_with_user_id=viewer_id, can_edit=False
        ))
        db.session.add(InternalShare(
            recording_id=rid, owner_id=owner_id, shared_with_user_id=editor_id, can_edit=True
        ))
        db.session.commit()

    try:
        with patch.object(app_module, "ENABLE_INTERNAL_SHARING", True):
            # View-only recipient must be rejected with 403
            r_view = _client_for(viewer_id).post(
                f"/api/v1/recordings/{rid}/tags", json={"tag_ids": [tid]}
            )
            assert r_view.status_code == 403, r_view.data

            # Shared editor who is NOT owner and NOT group admin cannot attach group tag
            r_edit = _client_for(editor_id).post(
                f"/api/v1/recordings/{rid}/tags", json={"tag_ids": [tid]}
            )
            assert r_edit.status_code == 200, r_edit.data
            body = r_edit.get_json()
            assert body["added_tags"] == []
            assert body["errors"]

            with _db():
                assert RecordingTag.query.filter_by(recording_id=rid, tag_id=tid).first() is None

            # Recording owner who is a group member can attach the group tag
            r_own = _client_for(owner_id).post(
                f"/api/v1/recordings/{rid}/tags", json={"tag_ids": [tid]}
            )
            assert r_own.status_code == 200, r_own.data
            assert len(r_own.get_json()["added_tags"]) == 1
    finally:
        _cleanup(user_ids=[owner_id, viewer_id, editor_id], recording_ids=[rid], tag_ids=[tid], group_ids=[gid])


# ---------------------------------------------------------------------------
# API v1 single & batch deletion uses StorageService and cleanup helpers
# ---------------------------------------------------------------------------

def test_api_v1_single_and_batch_delete_use_storage_service_and_cleanup():
    with _db():
        uid = _mk_user("del")
        rec1 = Recording(user_id=uid, title="r1", audio_path="local://abc/one.mp3", status="COMPLETED")
        rec2 = Recording(user_id=uid, title="r2", audio_path="s3://bucket/two.mp3", status="COMPLETED")
        db.session.add_all([rec1, rec2])
        db.session.flush()
        rid1, rid2 = rec1.id, rec2.id
        db.session.add(ProcessingJob(user_id=uid, recording_id=rid1, job_type="transcribe", status="completed"))
        db.session.add(ProcessingJob(user_id=uid, recording_id=rid2, job_type="transcribe", status="completed"))
        db.session.commit()

    mock_storage = MagicMock()
    mock_export = MagicMock()
    mock_speakers = MagicMock(return_value={"speakers_deleted": 0})
    try:
        with patch("src.api.api_v1.get_storage_service", return_value=mock_storage),              patch("src.api.api_v1.mark_export_as_deleted", mock_export),              patch("src.services.speaker_cleanup.cleanup_orphaned_speakers", mock_speakers):
            r1 = _client_for(uid).delete(f"/api/v1/recordings/{rid1}")
            assert r1.status_code == 200, r1.data
            mock_storage.delete.assert_called_with("local://abc/one.mp3", missing_ok=True)
            mock_export.assert_called_with(rid1)
            assert mock_speakers.call_count == 1

            r2 = _client_for(uid).delete("/api/v1/recordings/batch", json={"recording_ids": [rid2]})
            assert r2.status_code == 200, r2.data
            assert r2.get_json()["deleted"] == 1
            mock_storage.delete.assert_called_with("s3://bucket/two.mp3", missing_ok=True)
            mock_export.assert_called_with(rid2)
            assert mock_speakers.call_count == 2

        with _db():
            assert db.session.get(Recording, rid1) is None
            assert db.session.get(Recording, rid2) is None
    finally:
        _cleanup(user_ids=[uid], recording_ids=[rid1, rid2])


# ---------------------------------------------------------------------------
# API v1 chat passes operation_type & LLM helpers enforce budget without op type
# ---------------------------------------------------------------------------

def test_api_v1_chat_passes_operation_type_and_returns_429_on_budget_exceeded():
    with _db():
        uid = _mk_user("chatb")
        rec = Recording(user_id=uid, title="chat rec", transcription="hello world", status="COMPLETED")
        db.session.add(rec)
        db.session.commit()
        rid = rec.id

    try:
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "answer"

        with patch("src.services.llm.chat_client", MagicMock()),              patch("src.services.llm.call_chat_completion", return_value=mock_completion) as mock_call:
            resp = _client_for(uid).post(f"/api/v1/recordings/{rid}/chat", json={"message": "hi"})
            assert resp.status_code == 200, resp.data
            assert mock_call.call_args.kwargs.get("operation_type") == "chat"

        with patch("src.services.llm.chat_client", MagicMock()),              patch("src.services.llm.call_chat_completion", side_effect=TokenBudgetExceeded("Monthly limit reached", 100.0)):
            resp429 = _client_for(uid).post(f"/api/v1/recordings/{rid}/chat", json={"message": "hi"})
            assert resp429.status_code == 429, resp429.data
            assert resp429.get_json().get("budget_exceeded") is True
    finally:
        _cleanup(user_ids=[uid], recording_ids=[rid])


def test_llm_completion_helpers_enforce_budget_even_without_operation_type(monkeypatch):
    fake_client = MagicMock()
    usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.0, prompt_tokens_details=None)
    comp = MagicMock(choices=[MagicMock(message=MagicMock(content="ok"), finish_reason="stop")], usage=usage)
    fake_client.chat.completions.create.return_value = comp

    monkeypatch.setattr(llm_module, "client", fake_client)
    monkeypatch.setattr(llm_module, "chat_client", fake_client)
    monkeypatch.setattr(llm_module, "TEXT_MODEL_API_KEY", "test-key")
    monkeypatch.setattr(llm_module, "TEXT_MODEL_NAME", "gpt-4o")
    monkeypatch.setattr(llm_module, "CHAT_MODEL_API_KEY", None)
    monkeypatch.setattr(llm_module, "CHAT_MODEL_NAME", None)

    tracker_exceeded = MagicMock()
    tracker_exceeded.check_budget.return_value = (False, 110.0, "over budget")
    with patch("src.services.token_tracking.token_tracker", tracker_exceeded):
        with pytest.raises(TokenBudgetExceeded):
            call_llm_completion([{"role": "user", "content": "q"}], user_id=777, operation_type=None)
        with pytest.raises(TokenBudgetExceeded):
            call_chat_completion([{"role": "user", "content": "q"}], user_id=777, operation_type=None)

    tracker_ok = MagicMock()
    tracker_ok.check_budget.return_value = (True, 10.0, None)
    with patch("src.services.token_tracking.token_tracker", tracker_ok):
        call_llm_completion([{"role": "user", "content": "q"}], user_id=777, operation_type=None)
        call_chat_completion([{"role": "user", "content": "q"}], user_id=777, operation_type=None)
    assert tracker_ok.record_usage.call_count == 2
    assert tracker_ok.record_usage.call_args_list[0].kwargs["operation_type"] == "unspecified"
    assert tracker_ok.record_usage.call_args_list[1].kwargs["operation_type"] == "unspecified"
