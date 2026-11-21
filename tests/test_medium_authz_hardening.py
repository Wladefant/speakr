"""Regression tests for event delete permissions, bulk-tags group membership,
and null-owner recording deletion guards.
"""

import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.app as app_module
from src.app import app, db
from src.models import Event, Group, GroupMembership, InternalShare, Recording, RecordingTag, Tag, User

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
    return u.id


def _client_for(user_id):
    c = app.test_client()
    with c.session_transaction() as s:
        s["_user_id"] = str(user_id)
        s["_fresh"] = True
    return c


def _cleanup(user_ids=(), rec_ids=(), tag_ids=(), group_ids=(), event_ids=()):
    with _db():
        for eid in event_ids:
            ev = db.session.get(Event, eid)
            if ev:
                db.session.delete(ev)
        for rid in rec_ids:
            RecordingTag.query.filter_by(recording_id=rid).delete()
            InternalShare.query.filter_by(recording_id=rid).delete()
            rec = db.session.get(Recording, rid)
            if rec:
                db.session.delete(rec)
        for tid in tag_ids:
            RecordingTag.query.filter_by(tag_id=tid).delete()
            t = db.session.get(Tag, tid)
            if t:
                db.session.delete(t)
        for gid in group_ids:
            GroupMembership.query.filter_by(group_id=gid).delete()
            g = db.session.get(Group, gid)
            if g:
                db.session.delete(g)
        for uid in user_ids:
            InternalShare.query.filter(
                (InternalShare.owner_id == uid) | (InternalShare.shared_with_user_id == uid)
            ).delete()
            u = db.session.get(User, uid)
            if u:
                db.session.delete(u)
        db.session.commit()


def test_delete_event_blocks_view_only_share_recipient_and_allows_editor():
    with _db():
        owner_id = _mk_user("ev_own")
        viewer_id = _mk_user("ev_view")
        editor_id = _mk_user("ev_edit")
        rec = Recording(user_id=owner_id, title="ev rec", status="COMPLETED")
        db.session.add(rec)
        db.session.flush()
        rid = rec.id
        ev = Event(recording_id=rid, title="Standup", start_datetime=datetime.utcnow())
        db.session.add(ev)
        db.session.flush()
        eid = ev.id
        db.session.add(InternalShare(recording_id=rid, owner_id=owner_id, shared_with_user_id=viewer_id, can_edit=False))
        db.session.add(InternalShare(recording_id=rid, owner_id=owner_id, shared_with_user_id=editor_id, can_edit=True))
        db.session.commit()

    try:
        with patch.object(app_module, "ENABLE_INTERNAL_SHARING", True):
            r_view = _client_for(viewer_id).delete(f"/api/event/{eid}")
            assert r_view.status_code == 403, r_view.data
            with _db():
                assert db.session.get(Event, eid) is not None

            r_edit = _client_for(editor_id).delete(f"/api/event/{eid}")
            assert r_edit.status_code == 200, r_edit.data
            with _db():
                assert db.session.get(Event, eid) is None
    finally:
        _cleanup(user_ids=[owner_id, viewer_id, editor_id], rec_ids=[rid], event_ids=[eid])


def test_bulk_update_tags_enforces_group_membership_and_role():
    with _db():
        owner_id = _mk_user("bt_own")
        outsider_id = _mk_user("bt_out")
        member_id = _mk_user("bt_mem")
        grp = Group(name=f"btg_{uuid.uuid4().hex[:6]}")
        db.session.add(grp)
        db.session.flush()
        gid = grp.id
        db.session.add(GroupMembership(group_id=gid, user_id=owner_id, role="admin"))
        db.session.add(GroupMembership(group_id=gid, user_id=member_id, role="member"))
        gtag = Tag(name="secret-grp", user_id=owner_id, group_id=gid)
        db.session.add(gtag)
        db.session.flush()
        tid = gtag.id

        out_rec = Recording(user_id=outsider_id, title="outsider", status="COMPLETED")
        own_rec = Recording(user_id=owner_id, title="owner", status="COMPLETED")
        db.session.add_all([out_rec, own_rec])
        db.session.flush()
        out_rid, own_rid = out_rec.id, own_rec.id
        db.session.add(InternalShare(recording_id=own_rid, owner_id=owner_id, shared_with_user_id=member_id, can_edit=True))
        db.session.commit()

    try:
        with patch.object(app_module, "ENABLE_INTERNAL_SHARING", True):
            # Non-member cannot use group tag at all (403)
            r_out = _client_for(outsider_id).post(
                "/api/recordings/bulk-tags",
                json={"recording_ids": [out_rid], "tag_id": tid, "action": "add"},
            )
            assert r_out.status_code == 403, r_out.data

            # Regular group member who does not own the shared recording cannot attach group tag
            r_mem = _client_for(member_id).post(
                "/api/recordings/bulk-tags",
                json={"recording_ids": [own_rid], "tag_id": tid, "action": "add"},
            )
            assert r_mem.status_code == 200, r_mem.data
            assert r_mem.get_json()["affected_ids"] == []
            with _db():
                assert RecordingTag.query.filter_by(recording_id=own_rid, tag_id=tid).first() is None
    finally:
        _cleanup(user_ids=[owner_id, outsider_id, member_id], rec_ids=[out_rid, own_rid], tag_ids=[tid], group_ids=[gid])


def test_null_user_id_recordings_cannot_be_deleted_by_non_admin():
    with _db():
        user_id = _mk_user("nul_usr", is_admin=False)
        admin_id = _mk_user("nul_adm", is_admin=True)
        r1 = Recording(user_id=None, title="orphan1", status="COMPLETED")
        r2 = Recording(user_id=None, title="orphan2", status="COMPLETED")
        db.session.add_all([r1, r2])
        db.session.commit()
        rid1, rid2 = r1.id, r2.id

    try:
        resp_single = _client_for(user_id).delete(f"/recording/{rid1}")
        assert resp_single.status_code == 403, resp_single.data

        resp_bulk = _client_for(user_id).delete("/api/recordings/bulk", json={"recording_ids": [rid2]})
        assert resp_bulk.status_code == 200, resp_bulk.data
        assert resp_bulk.get_json()["deleted_count"] == 0

        with _db():
            assert db.session.get(Recording, rid1) is not None
            assert db.session.get(Recording, rid2) is not None

        # Admin is still permitted to clean up legacy unowned recordings
        resp_adm = _client_for(admin_id).delete(f"/recording/{rid1}")
        assert resp_adm.status_code == 200, resp_adm.data
        with _db():
            assert db.session.get(Recording, rid1) is None
    finally:
        _cleanup(user_ids=[user_id, admin_id], rec_ids=[rid1, rid2])
