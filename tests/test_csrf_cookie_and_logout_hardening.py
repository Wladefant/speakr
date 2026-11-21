"""Regression tests for cookie-auth CSRF enforcement on API blueprints, share-target origin checks, and POST-only logout."""

import io
import uuid

from src.app import app
from src.database import db
from src.models import APIToken, User


def _mk_user(prefix):
    uid = uuid.uuid4().hex[:8]
    u = User(username=f"{prefix}_{uid}", email=f"{prefix}_{uid}@example.com", password="x", email_verified=True)
    db.session.add(u)
    db.session.commit()
    return u


def test_api_v1_and_webhooks_require_csrf_for_ambient_cookies_but_allow_api_tokens():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED", True)
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        with app.app_context():
            user = _mk_user("csrf_v1")
            uid = user.id
            from src.models.api_token import hash_token
            plaintext = uuid.uuid4().hex
            tok_obj = APIToken(user_id=uid, token_hash=hash_token(plaintext), name="ci-test-tok")
            db.session.add(tok_obj)
            db.session.commit()

        # 1. Cookie-authenticated client WITHOUT CSRF token -> rejected with 400 on api_v1 & webhooks
        cookie_client = app.test_client()
        with cookie_client.session_transaction() as s:
            s["_user_id"] = str(uid)
            s["_fresh"] = True

        r_v1 = cookie_client.post("/api/v1/tags", json={"name": "csrf-blocked"})
        assert r_v1.status_code == 400

        r_wh = cookie_client.post("/api/v1/webhooks", json={"url": "https://example.com/hook", "events": ["recording.completed"]})
        assert r_wh.status_code == 400

        # 2. Programmatic API token client WITHOUT CSRF token -> NOT blocked by CSRF (never 400 CSRF error)
        api_client = app.test_client()
        r_api = api_client.post(
            "/api/v1/tags",
            json={"name": f"tok-tag-{uuid.uuid4().hex[:6]}"},
            headers={"Authorization": f"Bearer {plaintext}"},
        )
        assert r_api.status_code in (200, 201), r_api.get_data(as_text=True)

        with app.app_context():
            from src.models import Tag
            Tag.query.filter_by(user_id=uid).delete()
            APIToken.query.filter_by(user_id=uid).delete()
            db.session.delete(db.session.get(User, uid))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_share_target_blocks_cross_site_posts_and_allows_native_share():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED", True)
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        with app.app_context():
            user = _mk_user("share_csrf")
            uid = user.id

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(uid)
            s["_fresh"] = True

        # Cross-site Sec-Fetch-Site -> 403
        r1 = client.post(
            "/share-target",
            data={"shared_audio": (io.BytesIO(b"abc"), "note.mp3")},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert r1.status_code == 403

        # Cross-origin Origin header -> 403
        r2 = client.post(
            "/share-target",
            data={"shared_audio": (io.BytesIO(b"abc"), "note.mp3")},
            headers={"Origin": "https://evil.example"},
        )
        assert r2.status_code == 403

        # Cross-origin Referer header -> 403
        r3 = client.post(
            "/share-target",
            data={"shared_audio": (io.BytesIO(b"abc"), "note.mp3")},
            headers={"Referer": "https://evil.example/form.html"},
        )
        assert r3.status_code == 403

        # Native share sheet (Sec-Fetch-Site: none, no file -> redirects to missing_file, NOT 403)
        r4 = client.post("/share-target", headers={"Sec-Fetch-Site": "none"})
        assert r4.status_code == 302
        assert "missing_file" in r4.headers["Location"]

        with app.app_context():
            db.session.delete(db.session.get(User, uid))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_logout_rejects_get_enforces_post_csrf_and_clears_sso_link_state():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED", True)
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        with app.app_context():
            user = _mk_user("logout_csrf")
            uid = user.id

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(uid)
            s["_fresh"] = True
            s["sso_link_user_id"] = uid
            s["sso_link_state"] = "pending-state"

        # GET /logout is rejected (405 Method Not Allowed) and leaves session intact
        r_get = client.get("/logout")
        assert r_get.status_code == 405

        # POST /logout without CSRF token is rejected (400)
        r_post_nocsrf = client.post("/logout")
        assert r_post_nocsrf.status_code == 400

        # POST /logout with CSRF disabled (or valid token) clears session + sso_link_* keys
        app.config["WTF_CSRF_ENABLED"] = False
        r_post_ok = client.post("/logout", follow_redirects=False)
        assert r_post_ok.status_code == 302
        with client.session_transaction() as s:
            assert "_user_id" not in s
            assert "sso_link_user_id" not in s
            assert "sso_link_state" not in s

        with app.app_context():
            db.session.delete(db.session.get(User, uid))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf
