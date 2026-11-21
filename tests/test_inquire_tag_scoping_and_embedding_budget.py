"""Regression tests for Inquire tag visibility scoping and API-mode embedding budget enforcement."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import app
from src.database import db
from src.models import (
    Group,
    GroupMembership,
    InternalShare,
    Recording,
    RecordingTag,
    Tag,
    TranscriptChunk,
    User,
)
from src.services import embeddings as emb_svc
from src.services.llm import TokenBudgetExceeded


def _mk_user(prefix):
    uid = uuid.uuid4().hex[:8]
    u = User(username=f"{prefix}_{uid}", email=f"{prefix}_{uid}@example.com", password="x", email_verified=True)
    db.session.add(u)
    db.session.commit()
    return u


def test_api_embed_enforces_token_budget_before_calling_provider():
    fake_client = MagicMock()
    fake_tracker = MagicMock()
    fake_tracker.check_budget.return_value = (False, 100.0, "Monthly token budget exceeded")

    with app.app_context():
        with patch.object(emb_svc, "get_embedding_api_client", return_value=fake_client),              patch("src.services.token_tracking.token_tracker", fake_tracker):
            with pytest.raises(TokenBudgetExceeded):
                emb_svc._api_embed(["hello world"], user_id=77)

    fake_tracker.check_budget.assert_called_once_with(77)
    fake_client.embeddings.create.assert_not_called()


def test_inquire_search_and_process_chunks_return_429_on_embedding_budget_exceeded():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            user = _mk_user("emb_budget")
            rec = Recording(
                user_id=user.id,
                title="Budget Rec",
                original_filename="b.mp3",
                filename="b.mp3",
                transcription="Sample transcript content for chunking.",
                status="COMPLETED",
            )
            db.session.add(rec)
            db.session.commit()
            user_id, rec_id = user.id, rec.id

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(user_id)
            s["_fresh"] = True

        fake_client = MagicMock()
        fake_tracker = MagicMock()
        fake_tracker.check_budget.return_value = (False, 100.0, "Budget exhausted")

        with patch("src.api.inquire.ENABLE_INQUIRE_MODE", True),              patch.object(emb_svc, "EMBEDDINGS_AVAILABLE", True),              patch.object(emb_svc, "USE_API_EMBEDDINGS", True),              patch.object(emb_svc, "cosine_similarity", MagicMock()),              patch.object(emb_svc, "get_embedding_api_client", return_value=fake_client),              patch("src.services.token_tracking.token_tracker", fake_tracker):
            r_search = client.post("/api/inquire/search", json={"query": "budget test"})
            assert r_search.status_code == 429
            assert r_search.get_json().get("budget_exceeded") is True

            r_proc = client.post(f"/api/recording/{rec_id}/process_chunks")
            assert r_proc.status_code == 429
            assert r_proc.get_json().get("budget_exceeded") is True

        fake_client.embeddings.create.assert_not_called()

        with app.app_context():
            TranscriptChunk.query.filter_by(recording_id=rec_id).delete()
            db.session.delete(db.session.get(Recording, rec_id))
            db.session.delete(db.session.get(User, user_id))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf


def test_inquire_search_cannot_probe_hidden_owner_tags_and_chat_does_not_leak_foreign_tag_names():
    saved_csrf = app.config.get("WTF_CSRF_ENABLED")
    app.config["WTF_CSRF_ENABLED"] = False
    try:
        with app.app_context():
            owner = _mk_user("owner")
            recipient = _mk_user("recip")
            rec = Recording(
                user_id=owner.id,
                title="Shared Strategy",
                original_filename="s.mp3",
                filename="s.mp3",
                transcription="Quarterly roadmap review discussion",
                status="COMPLETED",
            )
            db.session.add(rec)
            db.session.flush()

            secret_tag = Tag(name="TOP_SECRET_LAYOFFS_TAG", user_id=owner.id, group_id=None)
            db.session.add(secret_tag)
            db.session.flush()

            db.session.add(RecordingTag(recording_id=rec.id, tag_id=secret_tag.id))
            share = InternalShare(
                recording_id=rec.id,
                owner_id=owner.id,
                shared_with_user_id=recipient.id,
                permission_level="view",
            )
            db.session.add(share)
            vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            chunk = TranscriptChunk(
                recording_id=rec.id,
                user_id=owner.id,
                chunk_index=0,
                content="Quarterly roadmap review discussion",
                embedding=vec.tobytes(),
            )
            db.session.add(chunk)
            db.session.commit()
            owner_id, recipient_id, rec_id, secret_tag_id = owner.id, recipient.id, rec.id, secret_tag.id

            # 1. Recipient cannot probe owner's hidden private tag via basic_text_search_chunks or semantic_search_chunks
            with patch.object(emb_svc, "ENABLE_INTERNAL_SHARING", True):
                res_basic = emb_svc.basic_text_search_chunks(
                    recipient_id, "roadmap", filters={"tag_ids": [secret_tag_id]}, top_k=5
                )
                assert res_basic == []

                # Owner CAN still search by their own private tag
                res_owner = emb_svc.basic_text_search_chunks(
                    owner_id, "roadmap", filters={"tag_ids": [secret_tag_id]}, top_k=5
                )
                assert len(res_owner) == 1

                # Semantic search also blocks probing owner's private tag
                fake_model = MagicMock()
                fake_model.encode.return_value = [vec]
                with patch.object(emb_svc, "EMBEDDINGS_AVAILABLE", True),                      patch.object(emb_svc, "USE_API_EMBEDDINGS", False),                      patch.object(emb_svc, "get_embedding_model", return_value=fake_model):
                    res_sem = emb_svc.semantic_search_chunks(
                        recipient_id, "roadmap", filters={"tag_ids": [secret_tag_id]}, top_k=5
                    )
                    assert res_sem == []

                # get_visible_tags_for_user never returns another user's private tag
                assert emb_svc.get_visible_tags_for_user(recipient_id, [secret_tag_id]) == []

        # 2. POST /api/inquire/chat must NOT inject foreign tag name into LLM system prompt
        captured_prompts = []

        class _FakeStreamChunk:
            def __init__(self, text):
                self.choices = [MagicMock(delta=MagicMock(content=text))]
                self.usage = None

        def _fake_chat_create(**kwargs):
            captured_prompts.append(kwargs.get("messages", []))
            return iter([_FakeStreamChunk("ok")])

        fake_llm_client = MagicMock()
        fake_llm_client.chat.completions.create.side_effect = _fake_chat_create
        fake_router_resp = MagicMock()
        fake_router_resp.choices = [MagicMock(message=MagicMock(content="RAG"))]
        fake_enrich_resp = MagicMock()
        fake_enrich_resp.choices = [MagicMock(message=MagicMock(content="roadmap"))]

        client = app.test_client()
        with client.session_transaction() as s:
            s["_user_id"] = str(recipient_id)
            s["_fresh"] = True

        with patch("src.api.inquire.ENABLE_INQUIRE_MODE", True),              patch("src.api.inquire.client", fake_llm_client),              patch("src.api.inquire.chat_client", fake_llm_client),              patch("src.api.inquire.call_llm_completion", side_effect=[fake_router_resp, fake_enrich_resp]),              patch("src.api.inquire.semantic_search_chunks", return_value=[]):
            resp = client.post(
                "/api/inquire/chat",
                json={"message": "What is on the roadmap?", "filter_tags": [secret_tag_id]},
            )
            assert resp.status_code == 200
            body = resp.get_data(as_text=True)
            assert "TOP_SECRET_LAYOFFS_TAG" not in body

        flat_prompt = str(captured_prompts)
        assert "TOP_SECRET_LAYOFFS_TAG" not in flat_prompt

        with app.app_context():
            TranscriptChunk.query.filter_by(recording_id=rec_id).delete()
            InternalShare.query.filter_by(recording_id=rec_id).delete()
            RecordingTag.query.filter_by(recording_id=rec_id).delete()
            db.session.delete(db.session.get(Tag, secret_tag_id))
            db.session.delete(db.session.get(Recording, rec_id))
            db.session.delete(db.session.get(User, recipient_id))
            db.session.delete(db.session.get(User, owner_id))
            db.session.commit()
    finally:
        app.config["WTF_CSRF_ENABLED"] = saved_csrf
