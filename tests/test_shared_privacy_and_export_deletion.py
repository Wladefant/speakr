"""Regression tests for shared-recording metadata/notes isolation and auto-export deletion."""

import uuid
from unittest.mock import MagicMock, patch

from src.app import app
from src.database import db
from src.file_exporter import mark_export_as_deleted
from src.models import (
    Folder,
    InternalShare,
    Recording,
    RecordingTag,
    SharedRecordingState,
    Tag,
    User,
)
from src.services import recording_merge as merge_svc


def _mk_user(prefix):
    uid = uuid.uuid4().hex[:8]
    u = User(username=f"{prefix}_{uid}", email=f"{prefix}_{uid}@example.com", password="x", email_verified=True)
    db.session.add(u)
    db.session.commit()
    return u


def test_shared_recipient_cannot_read_owner_private_notes_tags_folder_or_duplicates():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            owner = _mk_user("priv_own")
            recip = _mk_user("priv_rec")

            secret_folder = Folder(name="Executive M&A Folder", user_id=owner.id)
            secret_tag = Tag(name="PRIVATE_BOARD_TAG", user_id=owner.id, group_id=None)
            db.session.add_all([secret_folder, secret_tag])
            db.session.flush()

            rec = Recording(
                user_id=owner.id,
                title="Shared Sync",
                original_filename="sync.mp3",
                transcription="Public sync transcript.",
                notes="OWNER_SECRET_COMPENSATION_NOTE",
                folder_id=secret_folder.id,
                file_hash="abc123dup",
                status="COMPLETED",
            )
            dup_rec = Recording(
                user_id=owner.id,
                title="Private Duplicate",
                original_filename="dup.mp3",
                file_hash="abc123dup",
                status="COMPLETED",
            )
            db.session.add_all([rec, dup_rec])
            db.session.flush()

            db.session.add(RecordingTag(recording_id=rec.id, tag_id=secret_tag.id))
            db.session.add(
                InternalShare(
                    recording_id=rec.id,
                    owner_id=owner.id,
                    shared_with_user_id=recip.id,
                    can_edit=False,
                )
            )
            db.session.add(
                SharedRecordingState(
                    recording_id=rec.id,
                    user_id=recip.id,
                    personal_notes="RECIPIENT_OWN_NOTE",
                )
            )
            db.session.commit()
            owner_id, recip_id, rec_id, dup_id = owner.id, recip.id, rec.id, dup_rec.id
            folder_id, tag_id = secret_folder.id, secret_tag.id

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(recip_id)
            s["_fresh"] = True

        with patch("src.app.ENABLE_INTERNAL_SHARING", True),              patch("src.api.recordings.ENABLE_INTERNAL_SHARING", True):
            r_det = client.get(f"/api/v1/recordings/{rec_id}")
            assert r_det.status_code == 200
            payload = r_det.get_json()
            assert payload["notes"] == "RECIPIENT_OWN_NOTE"
            assert payload["folder_id"] is None
            assert payload["folder"] is None
            assert payload["tags"] == []
            assert payload["duplicate_info"] is None

            r_notes = client.get(f"/api/v1/recordings/{rec_id}/notes")
            assert r_notes.status_code == 200
            assert r_notes.get_json()["notes"] == "RECIPIENT_OWN_NOTE"

            r_fold = client.get(f"/api/recordings?folder={folder_id}")
            assert r_fold.status_code == 200
            assert all(r["id"] != rec_id for r in r_fold.get_json().get("recordings", []))

            r_tag = client.get("/api/recordings?q=tag:PRIVATE_BOARD_TAG")
            assert r_tag.status_code == 200
            assert all(r["id"] != rec_id for r in r_tag.get_json().get("recordings", []))

            captured_v1 = []
            def _fake_chat(messages, **kwargs):
                captured_v1.append(messages)
                resp = MagicMock()
                resp.choices = [MagicMock(message=MagicMock(content="ok"))]
                return resp

            with patch("src.services.llm.call_chat_completion", side_effect=_fake_chat):
                r_chat = client.post(f"/api/v1/recordings/{rec_id}/chat", json={"message": "summarize notes"})
                assert r_chat.status_code == 200
            assert "OWNER_SECRET_COMPENSATION_NOTE" not in str(captured_v1)
            assert "RECIPIENT_OWN_NOTE" in str(captured_v1)

        with app.app_context():
            SharedRecordingState.query.filter_by(recording_id=rec_id).delete()
            InternalShare.query.filter_by(recording_id=rec_id).delete()
            RecordingTag.query.filter_by(recording_id=rec_id).delete()
            db.session.delete(db.session.get(Recording, rec_id))
            db.session.delete(db.session.get(Recording, dup_id))
            db.session.delete(db.session.get(Tag, tag_id))
            db.session.delete(db.session.get(Folder, folder_id))
            db.session.delete(db.session.get(User, recip_id))
            db.session.delete(db.session.get(User, owner_id))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_mark_export_as_deleted_unlinks_files_and_all_delete_paths_invoke_it(tmp_path):
    user_dir = tmp_path / "alice"
    user_dir.mkdir()
    export_file = user_dir / "recording_9991.md"
    legacy_file = user_dir / "[deleted]_recording_9991.md"
    export_file.write_text("# Private Transcript", encoding="utf-8")
    legacy_file.write_text("# Legacy Deleted Copy", encoding="utf-8")

    with patch("src.file_exporter.ENABLE_AUTO_EXPORT", True),          patch("src.file_exporter.AUTO_EXPORT_DIR", str(tmp_path)):
        removed = mark_export_as_deleted(9991)

    assert removed is not None
    assert not export_file.exists()
    assert not legacy_file.exists()
    assert list(user_dir.iterdir()) == []

    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            user = _mk_user("exp_del")
            r1 = Recording(user_id=user.id, title="R1", original_filename="r1.mp3", status="COMPLETED")
            r2 = Recording(user_id=user.id, title="R2", original_filename="r2.mp3", status="COMPLETED")
            db.session.add_all([r1, r2])
            db.session.commit()
            uid, r1_id, r2_id = user.id, r1.id, r2.id

            called = []
            with patch("src.file_exporter.mark_export_as_deleted", side_effect=lambda rid: called.append(rid)):
                merge_svc._delete_recording(r2, MagicMock())
                db.session.commit()
            assert r2_id in called

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(uid)
            s["_fresh"] = True

        called_bulk = []
        with patch("src.api.recordings.mark_export_as_deleted", side_effect=lambda rid: called_bulk.append(rid)):
            resp = client.delete("/api/recordings/bulk", json={"recording_ids": [r1_id]})
            assert resp.status_code == 200
        assert r1_id in called_bulk

        with app.app_context():
            db.session.delete(db.session.get(User, uid))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf
