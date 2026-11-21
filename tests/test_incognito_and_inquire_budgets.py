"""Regression tests for incognito and Inquire token/transcription budget enforcement."""

import io
import json
import os
import sys
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.api.inquire as inquire_module
import src.api.recordings as recordings_module
import src.tasks.processing as processing_module
from src.app import app, db
from src.models import User
from src.services.llm import TokenBudgetExceeded

app.config["WTF_CSRF_ENABLED"] = False


@contextmanager
def _db():
    with app.app_context():
        yield


def _mk_user(prefix, token_budget=None, transcription_budget=None):
    suffix = uuid.uuid4().hex[:8]
    u = User(
        username=f"{prefix}_{suffix}",
        email=f"{prefix}_{suffix}@local.test",
        password="x",
        monthly_token_budget=token_budget,
        monthly_transcription_budget=transcription_budget,
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


def _cleanup_user(uid):
    with _db():
        u = db.session.get(User, uid)
        if u:
            db.session.delete(u)
            db.session.commit()


def _sse_events(resp):
    out = []
    for line in resp.get_data(as_text=True).splitlines():
        if line.startswith("data: "):
            raw = line[6:].strip()
            if raw == "[DONE]":
                continue
            try:
                out.append(json.loads(raw))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Incognito transcription, title, summary, and chat budget enforcement
# ---------------------------------------------------------------------------

def test_incognito_upload_blocks_over_transcription_budget_and_records_usage():
    with _db():
        uid = _mk_user("inc_tr")

    try:
        # 1. Over transcription budget -> 429 before connector is invoked
        mock_tracker = MagicMock()
        mock_tracker.check_budget.return_value = (False, 100.0, "Transcription limit reached")
        mock_registry = MagicMock()

        with patch.object(recordings_module, "ENABLE_INCOGNITO_MODE", True),              patch.object(processing_module, "transcription_tracker", mock_tracker),              patch("src.services.transcription.get_registry", return_value=mock_registry):
            resp = _client_for(uid).post(
                "/api/recordings/incognito",
                data={"file": (io.BytesIO(b"fake-audio"), "clip.mp3")},
                content_type="multipart/form-data",
            )
            assert resp.status_code == 429, resp.data
            assert resp.get_json().get("budget_exceeded") is True
            mock_registry.get_active_connector.assert_not_called()

        # 2. Within budget -> calls connector, records aggregate duration, routes title via call_llm_completion
        mock_tracker_ok = MagicMock()
        mock_tracker_ok.check_budget.return_value = (True, 10.0, None)
        fake_connector = MagicMock()
        fake_connector.supports_diarization = False
        fake_connector.specifications = MagicMock()
        fake_connector.model = "whisper-1"
        fake_connector.transcribe.return_value = SimpleNamespace(
            segments=[], text="This is a transcribed incognito audio recording.", has_diarization=lambda: False
        )
        fake_reg = MagicMock()
        fake_reg.get_active_connector.return_value = fake_connector
        fake_chunker = MagicMock()
        fake_chunker.needs_chunking.return_value = False
        fake_chunker.get_audio_duration.return_value = 42.0
        fake_title_resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Quick Note"))])

        with patch.object(recordings_module, "ENABLE_INCOGNITO_MODE", True),              patch.object(processing_module, "transcription_tracker", mock_tracker_ok),              patch.object(processing_module, "chunking_service", fake_chunker),              patch.object(processing_module, "client", MagicMock()),              patch.object(processing_module, "call_llm_completion", return_value=fake_title_resp) as mock_llm,              patch("src.services.transcription.get_registry", return_value=fake_reg):
            resp2 = _client_for(uid).post(
                "/api/recordings/incognito",
                data={"file": (io.BytesIO(b"fake-audio"), "clip.mp3")},
                content_type="multipart/form-data",
            )
            assert resp2.status_code == 200, resp2.data
            mock_tracker_ok.record_usage.assert_called_once()
            assert mock_tracker_ok.record_usage.call_args.kwargs["user_id"] == uid
            assert mock_tracker_ok.record_usage.call_args.kwargs["audio_duration_seconds"] == 42
            mock_llm.assert_called_once()
            assert mock_llm.call_args.kwargs["user_id"] == uid
            assert mock_llm.call_args.kwargs["operation_type"] == "title"
    finally:
        _cleanup_user(uid)


def test_incognito_summary_and_chat_enforce_token_budget_and_record_usage():
    with _db():
        uid = _mk_user("inc_llm")

    try:
        # Summary over budget -> 429
        with patch.object(recordings_module, "ENABLE_INCOGNITO_MODE", True),              patch.object(recordings_module, "client", MagicMock()),              patch.object(processing_module, "client", MagicMock()),              patch.object(processing_module, "call_llm_completion", side_effect=TokenBudgetExceeded("Token limit", 100.0)):
            r_sum = _client_for(uid).post(
                "/api/recordings/incognito/summary",
                json={"transcription": "Some transcript text"},
            )
            assert r_sum.status_code == 429, r_sum.data
            assert r_sum.get_json().get("budget_exceeded") is True

        # Chat over budget -> 429 before opening stream
        tracker_over = MagicMock()
        tracker_over.check_budget.return_value = (False, 105.0, "Token limit")
        with patch.object(recordings_module, "ENABLE_INCOGNITO_MODE", True),              patch.object(recordings_module, "chat_client", MagicMock()),              patch("src.services.token_tracking.token_tracker", tracker_over):
            r_chat_429 = _client_for(uid).post(
                "/api/recordings/incognito/chat",
                json={"transcription": "hello", "message": "summarize"},
            )
            assert r_chat_429.status_code == 429, r_chat_429.data
            assert r_chat_429.get_json().get("budget_exceeded") is True

        # Chat within budget -> streams via call_chat_completion with max_tokens and records usage chunk
        tracker_ok = MagicMock()
        tracker_ok.check_budget.return_value = (True, 10.0, None)
        usage_obj = SimpleNamespace(prompt_tokens=20, completion_tokens=10, total_tokens=30, cost=0.0, prompt_tokens_details=None)
        fake_stream = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))], usage=None),
            SimpleNamespace(choices=[], usage=usage_obj),
        ]
        with patch.object(recordings_module, "ENABLE_INCOGNITO_MODE", True),              patch.object(recordings_module, "chat_client", MagicMock()),              patch("src.services.token_tracking.token_tracker", tracker_ok),              patch("src.services.llm.call_chat_completion", return_value=iter(fake_stream)) as mock_chat:
            r_chat_ok = _client_for(uid).post(
                "/api/recordings/incognito/chat",
                json={"transcription": "hello", "message": "hi"},
            )
            assert r_chat_ok.status_code == 200, r_chat_ok.data
            events = _sse_events(r_chat_ok)
            assert any(e.get("content") == "Hi" for e in events)
            assert mock_chat.call_args.kwargs.get("operation_type") == "chat"
            assert mock_chat.call_args.kwargs.get("max_tokens") is not None
            tracker_ok.record_usage.assert_called_once()
            assert tracker_ok.record_usage.call_args.kwargs["total_tokens"] == 30
    finally:
        _cleanup_user(uid)


# ---------------------------------------------------------------------------
# Inquire chat: up-front budget, non-swallowed TokenBudgetExceeded, RAG usage
# ---------------------------------------------------------------------------

def test_inquire_chat_does_not_swallow_budget_exceeded_in_router_or_enrichment():
    with _db():
        uid = _mk_user("inq_b")

    try:
        # 1. Up-front check_budget failure returns 429 before semantic search
        tracker_over = MagicMock()
        tracker_over.check_budget.return_value = (False, 100.0, "Budget exhausted")
        mock_search = MagicMock()
        with patch.object(inquire_module, "ENABLE_INQUIRE_MODE", True),              patch.object(inquire_module, "client", MagicMock()),              patch.object(inquire_module, "token_tracker", tracker_over),              patch.object(inquire_module, "semantic_search_chunks", mock_search):
            r429 = _client_for(uid).post("/api/inquire/chat", json={"message": "hello"})
            assert r429.status_code == 429, r429.data
            mock_search.assert_not_called()

        # 2. Router raising TokenBudgetExceeded must NOT fall through to semantic_search_chunks
        tracker_ok = MagicMock()
        tracker_ok.check_budget.return_value = (True, 50.0, None)
        mock_search2 = MagicMock()
        with patch.object(inquire_module, "ENABLE_INQUIRE_MODE", True),              patch.object(inquire_module, "client", MagicMock()),              patch.object(inquire_module, "token_tracker", tracker_ok),              patch.object(inquire_module, "call_llm_completion", side_effect=TokenBudgetExceeded("router over", 100.0)),              patch.object(inquire_module, "semantic_search_chunks", mock_search2):
            resp = _client_for(uid).post("/api/inquire/chat", json={"message": "find notes"})
            assert resp.status_code == 200
            events = _sse_events(resp)
            assert any(e.get("budget_exceeded") is True for e in events)
            mock_search2.assert_not_called()

        # 3. Enrichment raising TokenBudgetExceeded must NOT fall through to semantic_search_chunks
        def router_ok_then_enrich_over(messages, **kwargs):
            if kwargs.get("operation_type") == "query_routing":
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="RAG"))])
            raise TokenBudgetExceeded("enrichment over", 100.0)

        mock_search3 = MagicMock()
        with patch.object(inquire_module, "ENABLE_INQUIRE_MODE", True),              patch.object(inquire_module, "client", MagicMock()),              patch.object(inquire_module, "token_tracker", tracker_ok),              patch.object(inquire_module, "call_llm_completion", side_effect=router_ok_then_enrich_over),              patch.object(inquire_module, "semantic_search_chunks", mock_search3):
            resp2 = _client_for(uid).post("/api/inquire/chat", json={"message": "find notes"})
            assert resp2.status_code == 200
            events2 = _sse_events(resp2)
            assert any(e.get("budget_exceeded") is True for e in events2)
            mock_search3.assert_not_called()
    finally:
        _cleanup_user(uid)


def test_inquire_rag_stream_records_final_usage_chunk_with_empty_choices():
    with _db():
        uid = _mk_user("inq_rag")

    try:
        tracker = MagicMock()
        tracker.check_budget.return_value = (True, 20.0, None)
        usage_obj = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140, cost=0.002, prompt_tokens_details=None)
        fake_stream = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Answer"))], usage=None),
            SimpleNamespace(choices=[], usage=usage_obj),
        ]

        def llm_se(messages, **kwargs):
            if kwargs.get("operation_type") == "query_routing":
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="RAG"))])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='["term"]'))])

        with patch.object(inquire_module, "ENABLE_INQUIRE_MODE", True),              patch.object(inquire_module, "client", MagicMock()),              patch.object(inquire_module, "EMBEDDINGS_AVAILABLE", True),              patch.object(inquire_module, "token_tracker", tracker),              patch.object(inquire_module, "call_llm_completion", side_effect=llm_se),              patch.object(inquire_module, "semantic_search_chunks", return_value=[]),              patch.object(inquire_module, "call_chat_completion", return_value=iter(fake_stream)):
            resp = _client_for(uid).post("/api/inquire/chat", json={"message": "question"})
            assert resp.status_code == 200
            events = _sse_events(resp)
            assert any(e.get("delta") == "Answer" for e in events)
            tracker.record_usage.assert_called_once()
            assert tracker.record_usage.call_args.kwargs["user_id"] == uid
            assert tracker.record_usage.call_args.kwargs["total_tokens"] == 140
    finally:
        _cleanup_user(uid)


def test_incognito_transcription_records_canonical_provider_and_positive_fallback_duration(tmp_path):
    audio_file = tmp_path / "short.mp3"
    audio_file.write_bytes(b"\x00" * 32000)

    class _FakeWhisperConnector:
        PROVIDER_NAME = "openai_whisper"
        model = "whisper-1"
        specifications = MagicMock()
        supports_diarization = False

        def transcribe(self, req):
            resp = MagicMock()
            resp.text = "Short incognito transcript."
            resp.segments = []
            resp.duration = None
            resp.has_diarization.return_value = False
            return resp

    fake_registry = MagicMock()
    fake_registry.get_active_connector.return_value = _FakeWhisperConnector()
    fake_user = MagicMock(id=999, transcription_language=None)

    for probe_return, expected_min in [(0.4, 1), (None, 1)]:
        fake_tracker = MagicMock()
        fake_tracker.check_budget.return_value = (True, 10.0, None)
        fake_chunking = MagicMock()
        fake_chunking.get_audio_duration.return_value = probe_return
        fake_chunking.needs_chunking.return_value = False

        with app.app_context(), \
             patch("src.tasks.processing.transcription_tracker", fake_tracker), \
             patch("src.tasks.processing.chunking_service", fake_chunking), \
             patch("src.services.transcription.get_registry", return_value=fake_registry), \
             patch("src.tasks.processing.is_video_file", return_value=False), \
             patch("src.tasks.processing.needs_conversion_for_connector", return_value=(False, None)), \
             patch("src.tasks.processing._generate_incognito_title", return_value="Test"):
            out = processing_module.transcribe_incognito(str(audio_file), "short.mp3", user=fake_user)

        assert out["transcription"] == "Short incognito transcript."
        fake_tracker.record_usage.assert_called_once()
        kw = fake_tracker.record_usage.call_args.kwargs
        assert kw["connector_type"] == "openai_whisper"
        assert kw["model_name"] == "whisper-1"
        assert kw["audio_duration_seconds"] >= expected_min
