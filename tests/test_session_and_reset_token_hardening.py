"""Regression tests for revocable session tokens, single-use password reset tokens, and remember cookie defaults."""

import uuid

from src.app import app, bcrypt, load_user
from src.database import db
from src.models.user import User
from src.services.email import generate_password_reset_token, verify_reset_token


def _make_user(is_admin=False):
    uid = uuid.uuid4().hex[:8]
    user = User(
        username=f"sec_{uid}",
        email=f"sec_{uid}@example.com",
        password=bcrypt.generate_password_hash("InitialPass!123").decode("utf-8"),
        is_admin=is_admin,
        email_verified=True,
    )
    db.session.add(user)
    db.session.commit()
    return user


def test_remember_cookie_security_defaults():
    assert app.config["REMEMBER_COOKIE_HTTPONLY"] is True
    assert app.config["REMEMBER_COOKIE_SAMESITE"] == "Lax"
    assert app.config["REMEMBER_COOKIE_SECURE"] == app.config["SESSION_COOKIE_SECURE"]


def test_password_reset_token_is_single_use_and_revokes_sessions():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            user = _make_user()
            user_id = user.id
            user_email = user.email

        client_old = app.test_client()
        resp = client_old.post("/login", data={"email": user_email, "password": "InitialPass!123"})
        assert resp.status_code == 302
        with client_old.session_transaction() as s:
            old_cookie_uid = s.get("_user_id")
        assert old_cookie_uid and ":" in old_cookie_uid

        with app.app_context():
            u = db.session.get(User, user_id)
            token1 = generate_password_reset_token(u.id)
            u.password_reset_token = token1
            db.session.commit()
            assert verify_reset_token(token1) == user_id

            token2 = generate_password_reset_token(u.id)
            if token2 == token1:
                token2 = token1 + "x"
            u.password_reset_token = token2
            db.session.commit()
            assert verify_reset_token(token1) is None

            u.password_reset_token = token1
            db.session.commit()

        anon = app.test_client()
        r1 = anon.post(
            f"/reset-password/{token1}",
            data={"password": "NewStrongPass!456", "confirm_password": "NewStrongPass!456"},
            follow_redirects=False,
        )
        assert r1.status_code == 302
        assert "/login" in r1.headers["Location"]

        with app.app_context():
            assert verify_reset_token(token1) is None
            assert load_user(old_cookie_uid) is None
            assert load_user(str(user_id)) is None

        r2 = anon.post(
            f"/reset-password/{token1}",
            data={"password": "HijackPass!789", "confirm_password": "HijackPass!789"},
            follow_redirects=False,
        )
        assert r2.status_code == 302
        assert "/forgot-password" in r2.headers["Location"]

        r_old = client_old.get("/account", follow_redirects=False)
        assert r_old.status_code == 302
        assert "/login" in r_old.headers["Location"]

        with app.app_context():
            u = db.session.get(User, user_id)
            db.session.delete(u)
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_change_password_and_logout_rotate_session_token():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            user = _make_user()
            user_id = user.id
            user_email = user.email

        c1 = app.test_client()
        c2 = app.test_client()
        assert c1.post("/login", data={"email": user_email, "password": "InitialPass!123"}).status_code == 302
        assert c2.post("/login", data={"email": user_email, "password": "InitialPass!123"}).status_code == 302

        with c1.session_transaction() as s:
            tok_before = s.get("_user_id")

        r_chg = c1.post(
            "/change_password",
            data={
                "current_password": "InitialPass!123",
                "new_password": "ChangedPass!999",
                "confirm_password": "ChangedPass!999",
            },
            follow_redirects=False,
        )
        assert r_chg.status_code == 302
        with c1.session_transaction() as s:
            tok_after = s.get("_user_id")
        assert tok_after != tok_before

        r_c2 = c2.get("/account", follow_redirects=False)
        assert r_c2.status_code == 302
        assert "/login" in r_c2.headers["Location"]

        assert c1.get("/logout", follow_redirects=False).status_code == 302
        with app.app_context():
            assert load_user(tok_after) is None
            u = db.session.get(User, user_id)
            db.session.delete(u)
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_admin_password_update_rotates_target_session_token():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            admin = _make_user(is_admin=True)
            victim = _make_user(is_admin=False)
            admin_id, victim_id = admin.id, victim.id
            admin_email, victim_email = admin.email, victim.email

        c_victim = app.test_client()
        assert c_victim.post("/login", data={"email": victim_email, "password": "InitialPass!123"}).status_code == 302
        with c_victim.session_transaction() as s:
            victim_tok = s.get("_user_id")

        c_admin = app.test_client()
        assert c_admin.post("/login", data={"email": admin_email, "password": "InitialPass!123"}).status_code == 302
        r_upd = c_admin.put(
            f"/admin/users/{victim_id}",
            json={"password": "AdminResetPass!321"},
        )
        assert r_upd.status_code == 200

        with app.app_context():
            assert load_user(victim_tok) is None
            db.session.delete(db.session.get(User, victim_id))
            db.session.delete(db.session.get(User, admin_id))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf
