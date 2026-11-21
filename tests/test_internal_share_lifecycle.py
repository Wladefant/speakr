"""Regression tests for transitive InternalShare cascade revocation and
group-tag / group-folder share ACL lifecycle cleanup.
"""

import os
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.api.folders as folders_module
import src.api.groups as groups_module
import src.api.recordings as recordings_module
import src.api.shares as shares_module
import src.api.tags as tags_module
import src.app as app_module
import src.services.embeddings as embeddings_module
from src.app import app, db, has_recording_access
from src.models import Folder, Group, GroupMembership, InternalShare, Recording, RecordingTag, SystemSetting, Tag, User

app.config["WTF_CSRF_ENABLED"] = False


@contextmanager
def _db():
    with app.app_context():
        yield


@contextmanager
def _internal_enabled():
    with patch.object(app_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(shares_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(groups_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(folders_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(tags_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(recordings_module, "ENABLE_INTERNAL_SHARING", True),          patch.object(embeddings_module, "ENABLE_INTERNAL_SHARING", True):
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


def _cleanup(user_ids=(), rec_ids=(), group_ids=(), tag_ids=(), folder_ids=()):
    with _db():
        for rid in rec_ids:
            RecordingTag.query.filter_by(recording_id=rid).delete()
            InternalShare.query.filter_by(recording_id=rid).delete()
            r = db.session.get(Recording, rid)
            if r:
                db.session.delete(r)
        for tid in tag_ids:
            RecordingTag.query.filter_by(tag_id=tid).delete()
            t = db.session.get(Tag, tid)
            if t:
                db.session.delete(t)
        for fid in folder_ids:
            f = db.session.get(Folder, fid)
            if f:
                db.session.delete(f)
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


# ---------------------------------------------------------------------------
# Multi-hop cascade revocation & recording-owner direct revocation
# ---------------------------------------------------------------------------

def test_cascade_revocation_traverses_multiple_hops_and_preserves_independent_branch():
    """Revoking A->B in A->B->C->D revokes B->C and C->D, while leaving A->E->F intact."""
    with _db():
        a = _mk_user("cas_a")
        b = _mk_user("cas_b")
        c = _mk_user("cas_c")
        d = _mk_user("cas_d")
        e = _mk_user("cas_e")
        f = _mk_user("cas_f")
        rec = Recording(user_id=a, title="chain", status="COMPLETED")
        db.session.add(rec)
        db.session.flush()
        rid = rec.id

        s_ab = InternalShare(recording_id=rid, owner_id=a, shared_with_user_id=b, can_reshare=True)
        s_bc = InternalShare(recording_id=rid, owner_id=b, shared_with_user_id=c, can_reshare=True)
        s_cd = InternalShare(recording_id=rid, owner_id=c, shared_with_user_id=d, can_reshare=False)
        s_ae = InternalShare(recording_id=rid, owner_id=a, shared_with_user_id=e, can_reshare=True)
        s_ef = InternalShare(recording_id=rid, owner_id=e, shared_with_user_id=f, can_reshare=False)
        db.session.add_all([s_ab, s_bc, s_cd, s_ae, s_ef])
        db.session.commit()
        ab_id, bc_id, cd_id, ae_id, ef_id = s_ab.id, s_bc.id, s_cd.id, s_ae.id, s_ef.id

    try:
        with _internal_enabled():
            resp = _client_for(a).delete(f"/api/internal-shares/{ab_id}")
            assert resp.status_code == 200, resp.data
            body = resp.get_json()
            assert body["cascaded"] == 2
            assert body["revoked_count"] == 3

        with _db():
            assert db.session.get(InternalShare, ab_id) is None
            assert db.session.get(InternalShare, bc_id) is None
            assert db.session.get(InternalShare, cd_id) is None
            assert db.session.get(InternalShare, ae_id) is not None
            assert db.session.get(InternalShare, ef_id) is not None
    finally:
        _cleanup(user_ids=[a, b, c, d, e, f], rec_ids=[rid])


def test_recording_owner_can_revoke_downstream_reshared_child_share():
    """Recording owner A can directly revoke B->C (cascading to C->D) while keeping A->B."""
    with _db():
        a = _mk_user("ownrev_a")
        b = _mk_user("ownrev_b")
        c = _mk_user("ownrev_c")
        d = _mk_user("ownrev_d")
        rec = Recording(user_id=a, title="owner revoke", status="COMPLETED")
        db.session.add(rec)
        db.session.flush()
        rid = rec.id

        s_ab = InternalShare(recording_id=rid, owner_id=a, shared_with_user_id=b, can_reshare=True)
        s_bc = InternalShare(recording_id=rid, owner_id=b, shared_with_user_id=c, can_reshare=True)
        s_cd = InternalShare(recording_id=rid, owner_id=c, shared_with_user_id=d, can_reshare=False)
        db.session.add_all([s_ab, s_bc, s_cd])
        db.session.commit()
        ab_id, bc_id, cd_id = s_ab.id, s_bc.id, s_cd.id

    try:
        with _internal_enabled():
            resp = _client_for(a).delete(f"/api/internal-shares/{bc_id}")
            assert resp.status_code == 200, resp.data
            assert resp.get_json()["cascaded"] == 1

        with _db():
            assert db.session.get(InternalShare, ab_id) is not None
            assert db.session.get(InternalShare, bc_id) is None
            assert db.session.get(InternalShare, cd_id) is None
    finally:
        _cleanup(user_ids=[a, b, c, d], rec_ids=[rid])


# ---------------------------------------------------------------------------
# Group-tag & group-folder share ACL lifecycle and read-time membership checks
# ---------------------------------------------------------------------------

def test_group_member_removal_and_demotion_revoke_or_downgrade_derived_shares():
    with _db():
        admin_id = _mk_user("gacl_adm", is_admin=True)
        owner_id = _mk_user("gacl_own")
        member_id = _mk_user("gacl_mem")
        grp = Group(name=f"g_{uuid.uuid4().hex[:6]}")
        db.session.add(grp)
        db.session.flush()
        gid = grp.id

        db.session.add(GroupMembership(group_id=gid, user_id=owner_id, role="admin"))
        db.session.add(GroupMembership(group_id=gid, user_id=member_id, role="admin"))

        tag = Tag(
            name="team-tag", user_id=owner_id, group_id=gid,
            auto_share_on_apply=True, share_with_group_lead=True,
        )
        folder = Folder(
            name="team-folder", user_id=owner_id, group_id=gid,
            auto_share_on_apply=True, share_with_group_lead=True,
        )
        db.session.add_all([tag, folder])
        db.session.flush()
        tid, fid = tag.id, folder.id

        r_tag = Recording(user_id=owner_id, title="tag rec", status="COMPLETED")
        r_fld = Recording(user_id=owner_id, title="fld rec", folder_id=fid, status="COMPLETED")
        db.session.add_all([r_tag, r_fld])
        db.session.flush()
        rid_tag, rid_fld = r_tag.id, r_fld.id

        db.session.add(RecordingTag(recording_id=rid_tag, tag_id=tid, order=1))
        sh_tag = InternalShare(
            recording_id=rid_tag, owner_id=owner_id, shared_with_user_id=member_id,
            can_edit=True, source_type="group_tag", source_tag_id=tid,
        )
        sh_fld = InternalShare(
            recording_id=rid_fld, owner_id=owner_id, shared_with_user_id=member_id,
            can_edit=True, source_type="group_folder", source_folder_id=fid,
        )
        db.session.add_all([sh_tag, sh_fld])
        db.session.commit()

    try:
        with _internal_enabled():
            # 1. Demote member from admin -> member: loses edit access, keeps view
            r_demote = _client_for(admin_id).put(
                f"/api/admin/groups/{gid}/members/{member_id}", json={"role": "member"}
            )
            assert r_demote.status_code == 200, r_demote.data
            with _db():
                mem = db.session.get(User, member_id)
                assert has_recording_access(db.session.get(Recording, rid_tag), mem, require_edit=False) is True
                assert has_recording_access(db.session.get(Recording, rid_tag), mem, require_edit=True) is False
                assert has_recording_access(db.session.get(Recording, rid_fld), mem, require_edit=True) is False

            # 2. Remove member from group: loses ALL access across recording & inquire helpers
            r_rem = _client_for(admin_id).delete(f"/api/admin/groups/{gid}/members/{member_id}")
            assert r_rem.status_code == 200, r_rem.data
            with _db():
                mem = db.session.get(User, member_id)
                assert has_recording_access(db.session.get(Recording, rid_tag), mem) is False
                assert has_recording_access(db.session.get(Recording, rid_fld), mem) is False
                assert rid_tag not in embeddings_module.get_accessible_recording_ids(member_id)
                assert rid_fld not in embeddings_module.get_accessible_recording_ids(member_id)
                assert rid_tag not in recordings_module.get_accessible_recording_ids(member_id)
    finally:
        _cleanup(user_ids=[admin_id, owner_id, member_id], rec_ids=[rid_tag, rid_fld], group_ids=[gid], tag_ids=[tid], folder_ids=[fid])


def test_folder_removal_and_stale_group_share_read_time_rejection():
    with _db():
        owner_id = _mk_user("frem_own")
        member_id = _mk_user("frem_mem")
        grp = Group(name=f"gf_{uuid.uuid4().hex[:6]}")
        db.session.add(grp)
        db.session.flush()
        gid = grp.id
        db.session.add(GroupMembership(group_id=gid, user_id=owner_id, role="admin"))
        db.session.add(GroupMembership(group_id=gid, user_id=member_id, role="member"))

        folder = Folder(name="shared-f", user_id=owner_id, group_id=gid, auto_share_on_apply=True)
        db.session.add(folder)
        db.session.flush()
        fid = folder.id

        rec = Recording(user_id=owner_id, title="in folder", folder_id=fid, status="COMPLETED")
        db.session.add(rec)
        db.session.flush()
        rid = rec.id

        # Create a stale orphan group_tag share (source_tag_id=None, simulating deleted tag)
        rec_stale = Recording(user_id=owner_id, title="stale tag rec", status="COMPLETED")
        db.session.add(rec_stale)
        db.session.flush()
        rid_stale = rec_stale.id

        db.session.add(InternalShare(
            recording_id=rid, owner_id=owner_id, shared_with_user_id=member_id,
            can_edit=False, source_type="group_folder", source_folder_id=fid,
        ))
        db.session.add(InternalShare(
            recording_id=rid_stale, owner_id=owner_id, shared_with_user_id=member_id,
            can_edit=True, source_type="group_tag", source_tag_id=None,
        ))
        db.session.commit()

    try:
        with _internal_enabled(), patch.object(SystemSetting, "get_setting", return_value=True):
            # Read-time guard immediately rejects stale group_tag share with source_tag_id=None
            with _db():
                mem = db.session.get(User, member_id)
                assert has_recording_access(db.session.get(Recording, rid_stale), mem) is False
                assert rid_stale not in embeddings_module.get_accessible_recording_ids(member_id)

            # Removing recording from folder revokes the group_folder share
            resp = _client_for(owner_id).delete(f"/api/recordings/{rid}/folder")
            assert resp.status_code == 200, resp.data
            with _db():
                mem = db.session.get(User, member_id)
                assert has_recording_access(db.session.get(Recording, rid), mem) is False
                assert rid not in embeddings_module.get_accessible_recording_ids(member_id)
    finally:
        _cleanup(user_ids=[owner_id, member_id], rec_ids=[rid, rid_stale], group_ids=[gid], folder_ids=[fid])
