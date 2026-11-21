"""Regression tests for complete data cleanup during admin user deletion."""

import os
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app, db
from src.models import Event, ProcessingJob, Recording, RecordingSession, TranscriptChunk, User

app.config["WTF_CSRF_ENABLED"] = False


@contextmanager
def _db():
    with app.app_context():
        yield


def _mk_user(prefix, is_admin=False):
    suffix = uuid.uuid4().hex[:8]
    u = User(
        username=f"{prefix}_{suffix}",
        email=f"{prefix}_{suffix}@local.test",
        password="x",
        is_admin=is_admin,
    )
    db.session.add(u)
    db.session.commit()
    return u.id, u.username


def test_admin_delete_user_removes_recordings_children_sessions_and_exports():
    with _db():
        admin_id, _ = _mk_user("deladm", is_admin=True)
        target_id, target_username = _mk_user("delusr", is_admin=False)
        editor_id, _ = _mk_user("deledt", is_admin=False)

        rec = Recording(
            user_id=target_id,
            title="private recording",
            transcription="confidential transcript",
            summary="confidential summary",
            notes="private notes",
            status="COMPLETED",
        )
        db.session.add(rec)
        db.session.flush()
        rec_id = rec.id

        ev = Event(
            recording_id=rec_id,
            title="Private Meeting",
            start_datetime=datetime.utcnow(),
        )
        chunk = TranscriptChunk(
            recording_id=rec_id,
            user_id=target_id,
            chunk_index=0,
            content="confidential chunk",
        )
        # Shared-editor job attached to target's recording (user_id != target_id)
        shared_job = ProcessingJob(
            user_id=editor_id,
            recording_id=rec_id,
            job_type="summarize",
            status="completed",
        )
        sess_id = uuid.uuid4().hex
        rec_sess = RecordingSession(
            id=sess_id,
            user_id=target_id,
            status="recording",
            mime_type="audio/webm",
        )
        db.session.add_all([ev, chunk, shared_job, rec_sess])
        db.session.commit()
        ev_id = ev.id
        chunk_id = chunk.id
        job_id = shared_job.id

    with tempfile.TemporaryDirectory(prefix="speakr_del_test_") as tmpdir:
        upload_dir = Path(tmpdir) / "uploads"
        session_dir = upload_dir / "_sessions" / sess_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "chunk_0.webm").write_bytes(b"audio-chunk")

        export_root = Path(tmpdir) / "exports"
        user_export_dir = export_root / target_username
        user_export_dir.mkdir(parents=True, exist_ok=True)
        (user_export_dir / f"recording_{rec_id}.md").write_text("# Private", encoding="utf-8")

        try:
            client = app.test_client()
            with client.session_transaction() as s:
                s["_user_id"] = str(admin_id)
                s["_fresh"] = True

            with patch.dict(app.config, {"UPLOAD_FOLDER": str(upload_dir)}),                  patch("src.file_exporter.AUTO_EXPORT_DIR", str(export_root)):
                resp = client.delete(f"/admin/users/{target_id}")
                assert resp.status_code == 200, resp.data
                assert resp.get_json().get("success") is True

            with _db():
                assert db.session.get(User, target_id) is None
                assert db.session.get(Recording, rec_id) is None
                assert db.session.get(Event, ev_id) is None
                assert db.session.get(TranscriptChunk, chunk_id) is None
                assert db.session.get(ProcessingJob, job_id) is None
                assert db.session.get(RecordingSession, sess_id) is None

            assert not session_dir.exists(), "Orphaned recording session directory survived user deletion"
            assert not user_export_dir.exists(), "User auto-export directory survived user deletion"
        finally:
            with _db():
                for uid in (admin_id, editor_id, target_id):
                    u = db.session.get(User, uid)
                    if u:
                        db.session.delete(u)
                db.session.commit()
