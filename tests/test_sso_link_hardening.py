"""Regression tests for SSO account-linking session binding and validation."""

import os
import sys
import uuid
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import redirect
from src.app import app, db
from src.models import User

app.config["WTF_CSRF_ENABLED"] = False


@contextmanager
def _db():
    with app.app_context():
        yield


def _mk_user(prefix, sso_subject=None):
    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"{prefix}_{suffix}",
        email=f"{prefix}_{suffix}@local.test",
        password="hashed-password",
        sso_subject=sso_subject,
    )
    db.session.add(user)
    db.session.commit()
    return user.id


def _cleanup_users(*user_ids):
    with _db():
        for uid in user_ids:
            u = db.session.get(User, uid)
            if u:
                db.session.delete(u)
        db.session.commit()


def _fake_oauth(userinfo):
    oauth = MagicMock()
    oauth.sso.authorize_redirect.side_effect = (
        lambda redirect_uri=None, state=None: redirect(f"https://idp.example/auth?state={state or ''}")
    )
    oauth.sso.authorize_access_token.return_value = {"userinfo": userinfo}
    return oauth


def test_logout_and_direct_sso_login_clear_pending_sso_link_state():
    with _db():
        uid = _mk_user("ssoclr")

    try:
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["_user_id"] = str(uid)
            sess["_fresh"] = True
            sess["sso_link_user_id"] = uid
            sess["sso_link_state"] = "state-abc"

        resp = client.get("/logout")
        assert resp.status_code == 302
        with client.session_transaction() as sess:
            assert "sso_link_user_id" not in sess
            assert "sso_link_state" not in sess

        # Direct /auth/sso/login must also clear stale link state
        with client.session_transaction() as sess:
            sess["sso_link_user_id"] = uid
            sess["sso_link_state"] = "stale-state"

        fake = _fake_oauth({"sub": "sub-x", "email": "x@example.com"})
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=fake),              patch("src.api.auth.get_sso_config", return_value={"redirect_uri": "https://app/cb", "provider_name": "OIDC"}):
            resp2 = client.get("/auth/sso/login")
            assert resp2.status_code == 302
        with client.session_transaction() as sess:
            assert "sso_link_user_id" not in sess
            assert "sso_link_state" not in sess
    finally:
        _cleanup_users(uid)


def test_sso_callback_rejects_unauthenticated_mismatched_user_and_bad_state():
    with _db():
        victim_id = _mk_user("ssovic")
        other_id = _mk_user("ssooth")

    sub_attacker = f"sub-att-{uuid.uuid4().hex[:8]}"
    fake = _fake_oauth({"sub": sub_attacker, "email": "att@example.com"})
    cfg = {"redirect_uri": "https://app/cb", "provider_name": "OIDC"}

    try:
        # 1. Unauthenticated callback with stale sso_link_user_id
        c1 = app.test_client()
        with c1.session_transaction() as sess:
            sess["sso_link_user_id"] = victim_id
            sess["sso_link_state"] = "good-state"
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=fake),              patch("src.api.auth.get_sso_config", return_value=cfg):
            r1 = c1.get("/auth/sso/callback?state=good-state")
            assert r1.status_code == 302

        # 2. Authenticated as DIFFERENT user with stale sso_link_user_id
        c2 = app.test_client()
        with c2.session_transaction() as sess:
            sess["_user_id"] = str(other_id)
            sess["_fresh"] = True
            sess["sso_link_user_id"] = victim_id
            sess["sso_link_state"] = "good-state"
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=fake),              patch("src.api.auth.get_sso_config", return_value=cfg):
            r2 = c2.get("/auth/sso/callback?state=good-state")
            assert r2.status_code == 302

        # 3. Authenticated as victim, but OAuth state does not match link state
        c3 = app.test_client()
        with c3.session_transaction() as sess:
            sess["_user_id"] = str(victim_id)
            sess["_fresh"] = True
            sess["sso_link_user_id"] = victim_id
            sess["sso_link_state"] = "expected-state"
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=fake),              patch("src.api.auth.get_sso_config", return_value=cfg):
            r3 = c3.get("/auth/sso/callback?state=wrong-state")
            assert r3.status_code == 302

        with _db():
            victim = db.session.get(User, victim_id)
            assert victim.sso_subject is None
    finally:
        _cleanup_users(victim_id, other_id)


def test_sso_callback_does_not_overwrite_existing_sso_subject_and_allows_valid_link():
    existing_sub = f"sub-exist-{uuid.uuid4().hex[:8]}"
    new_sub = f"sub-new-{uuid.uuid4().hex[:8]}"
    with _db():
        linked_id = _mk_user("ssolnk", sso_subject=existing_sub)
        fresh_id = _mk_user("ssofrsh")

    cfg = {"redirect_uri": "https://app/cb", "provider_name": "OIDC"}
    try:
        # Attempt to overwrite an already-linked account via forged session state
        c_linked = app.test_client()
        with c_linked.session_transaction() as sess:
            sess["_user_id"] = str(linked_id)
            sess["_fresh"] = True
            sess["sso_link_user_id"] = linked_id
            sess["sso_link_state"] = "st-1"
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=_fake_oauth({"sub": new_sub})),              patch("src.api.auth.get_sso_config", return_value=cfg):
            r_ow = c_linked.get("/auth/sso/callback?state=st-1")
            assert r_ow.status_code == 302
        with _db():
            assert db.session.get(User, linked_id).sso_subject == existing_sub

        # Valid end-to-end link flow for an unlinked authenticated user
        c_fresh = app.test_client()
        with c_fresh.session_transaction() as sess:
            sess["_user_id"] = str(fresh_id)
            sess["_fresh"] = True
        fake = _fake_oauth({"sub": new_sub, "name": "Fresh User"})
        with patch("src.api.auth.is_sso_enabled", return_value=True),              patch("src.api.auth.get_sso_client", return_value=fake),              patch("src.api.auth.get_sso_config", return_value=cfg):
            r_start = c_fresh.post("/auth/sso/link")
            assert r_start.status_code == 302
            with c_fresh.session_transaction() as sess:
                bound_state = sess.get("sso_link_state")
                assert bound_state
                assert sess.get("sso_link_user_id") == fresh_id
            r_cb = c_fresh.get(f"/auth/sso/callback?state={bound_state}")
            assert r_cb.status_code == 302

        with _db():
            fresh = db.session.get(User, fresh_id)
            assert fresh.sso_subject == new_sub
            assert fresh.sso_provider == "OIDC"
    finally:
        _cleanup_users(linked_id, fresh_id)
