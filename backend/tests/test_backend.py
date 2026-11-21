from contextlib import contextmanager
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Retriever
from app.embeddings import Embeddings
from app.errors import ServiceError
from app.generation import Generator, build_context
from app.main import create_app
from app.schemas import Chunk

TOKEN = "test-token-with-at-least-24-characters"


def settings(**changes):
    return Settings(_env_file=None, dev_auth_enabled=True, dev_api_token=TOKEN, **changes)


def chunk(**changes):
    return Chunk(**({"chunk_id": 2, "document_id": 1, "title": "Документ",
                    "url": "https://example.com", "content": "Резервные копии ежедневно",
                    "rrf_score": .03} | changes))


def services():
    return SimpleNamespace(embeddings=Mock(), retriever=Mock(), generator=Mock())


def test_auth_validation_and_no_client_identity_override():
    svc = services()
    svc.retriever.search.return_value = [chunk()]
    cfg = settings(dev_user_id=7)
    with TestClient(create_app(cfg, svc)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/search", json={"query": "тест"}).status_code == 401
        svc.embeddings.encode.assert_not_called()
        response = client.post("/search", headers={"Authorization": f"Bearer {TOKEN}"},
                               json={"query": "тест", "user_id": 999, "roles": ["admin"]})
        assert response.status_code == 200
        assert svc.retriever.search.call_args.args[0] == 7
        assert response.json()["results"][0]["chunk_id"] == 2
        assert client.post("/search", headers={"Authorization": f"Bearer {TOKEN}"},
                           json={"query": " ", "limit": 100}).status_code == 422


def test_missing_embedding_configuration_is_explicit():
    cfg = settings()
    with httpx.Client() as http:
        svc = services()
        svc.embeddings = Embeddings(cfg, http)
        with TestClient(create_app(cfg, svc)) as client:
            response = client.post("/ask", headers={"Authorization": f"Bearer {TOKEN}"},
                                   json={"question": "Вопрос"})
            assert response.status_code == 503
            assert "EMBEDDING_BASE_URL" in response.json()["detail"]
            svc.retriever.search.assert_not_called()
            svc.generator.answer.assert_not_called()


@pytest.mark.parametrize("vector", [[0.] * 1024, [1.] * 768, [float('nan')] * 1024])
def test_embedding_rejects_invalid_vectors(vector):
    # NaN is supplied through a mock response, not invalid JSON serialization.
    http = Mock()
    http.post.return_value.json.return_value = {"data": [{"embedding": vector}]}
    service = Embeddings(settings(embedding_base_url="http://embedding/v1"), http)
    with pytest.raises(ServiceError):
        service.encode("вопрос")


def test_embedding_api_contract():
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"data": [{"embedding": [1.] * 1024}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        vector = Embeddings(settings(embedding_base_url="http://embedding/v1"), http).encode("тест")
        assert vector.shape == (1024,)


def test_empty_context_does_not_call_llm():
    http = Mock()
    result = Generator(settings(), http).answer("вопрос", [])
    assert result.sources == []
    http.post.assert_not_called()


@pytest.mark.parametrize("provider,path,data", [
    ("ollama", "/api/chat", {"message": {"content": "Ежедневно [1]"}}),
    ("openai", "/chat/completions", {"choices": [{"message": {"content": "Ежедневно [1]"}}]}),
    ("lmstudio", "/chat/completions", {"choices": [{"message": {"content": "Ежедневно [1]"}}]}),
])
def test_llm_providers(provider, path, data):
    def handler(request):
        assert request.url.path.endswith(path)
        return httpx.Response(200, json=data)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        result = Generator(settings(llm_provider=provider), http).answer("Когда?", [chunk()])
        assert result.sources[0].chunk_id == 2


def test_unknown_citation_is_not_returned():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"message": {"content": "Ответ [99]"}}
    ))) as http:
        with pytest.raises(ServiceError):
            Generator(settings(), http).answer("вопрос", [chunk()])


@pytest.mark.parametrize("status,hint", [(400, "LLM_MODEL"), (401, "LLM_API_KEY"),
                                       (404, "/v1/models"), (500, "server logs")])
def test_llm_http_error_has_actionable_detail(status, hint):
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        status, json={"error": "private upstream details"}
    ))) as http:
        with pytest.raises(ServiceError) as error:
            Generator(settings(llm_provider="lmstudio"), http).answer("вопрос", [chunk()])
        assert hint in str(error.value)
        assert "private upstream" not in str(error.value)


def test_context_budget_excludes_unused_sources():
    text, sources = build_context([chunk(content="a" * 10000)] * 10)
    assert len(text) < 14500
    assert 0 < len(sources) < 10


def test_reasoning_without_final_answer_has_specific_error():
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"choices": [{"message": {"content": "", "reasoning_content": "internal reasoning"}}]}
    ))) as http:
        with pytest.raises(ServiceError, match="reasoning being disabled"):
            Generator(settings(llm_provider="lmstudio"), http).answer("вопрос", [chunk()])


@pytest.mark.parametrize("provider", ["lmstudio", "ollama", "openai"])
def test_reasoning_cannot_be_enabled(provider):
    cfg=settings(llm_provider=provider,llm_model="qwen/qwen3.5-9b",llm_thinking=True,deep_llm_thinking=True)
    _,payload=Generator(cfg,Mock()).chat_payload([{"role":"user","content":"test"}],100)
    assert not hasattr(cfg,'llm_thinking')
    if provider=='lmstudio':assert payload['reasoning_effort']=='none'
    elif provider=='ollama':assert payload['think'] is False
    else:assert payload['chat_template_kwargs']['enable_thinking'] is False


def test_dev_mode_cannot_be_enabled_in_production():
    with pytest.raises(ValueError):
        settings(app_env="production")


@pytest.mark.parametrize("provider", ["lmstudio", "ollama"])
def test_stream_deltas_and_citations(provider):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["stream"] is True
        if provider == "lmstudio":
            assert payload["reasoning_effort"] == "none"
            records = [
                {"choices": [{"delta": {"reasoning_content": "private"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "Ответ "}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "[1]"}, "finish_reason": "stop"}]},
            ]
            body = ''.join('data: ' + json.dumps(item) + '\n\n' for item in records)
        else:
            body = '\n'.join(json.dumps(item) for item in [
                {"message": {"content": "Ответ "}, "done": False},
                {"message": {"content": "[1]"}, "done": True, "done_reason": "stop"},
            ])
        return httpx.Response(200, text=body)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        events = list(Generator(settings(llm_provider=provider), http).stream_answer("тест", [chunk()]))
    assert [event for event, data in events] == ["sources", "delta", "delta", "done"]
    assert events[-1][1]["answer"] == "Ответ [1]"
    assert "private" not in str(events)


@pytest.mark.parametrize("finish,content", [(None, "Ответ [1]"), ("length", "Ответ [1]"), ("stop", "Ответ [99]")])
def test_incomplete_or_invalid_stream_is_not_success(finish, content):
    body = 'data: ' + json.dumps({"choices": [{"delta": {"content": content}, "finish_reason": finish}]}) + '\n\n'
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))) as http:
        with pytest.raises(ServiceError):
            list(Generator(settings(llm_provider="lmstudio"), http).stream_answer("тест", [chunk()]))


def test_stream_endpoint_auth_and_error_events():
    svc = services()
    svc.retriever.search.return_value = [chunk()]
    def broken_stream(*args):
        yield "delta", {"text": "черновик"}
        raise ServiceError("LLM stream timed out", 504)
    svc.generator.stream_answer.side_effect = broken_stream
    with TestClient(create_app(settings(), svc)) as client:
        assert client.post("/ask/stream", json={"question": "тест"}).status_code == 401
        response = client.post("/ask/stream", json={"question": "тест"},
                               headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: delta" in response.text and "event: error" in response.text
    assert "event: done" not in response.text


def test_empty_stream_does_not_call_model():
    http = Mock()
    events = list(Generator(settings(), http).stream_answer("тест", []))
    assert events[0][0] == "done"
    http.stream.assert_not_called()


def test_database_identity_is_transaction_scoped_and_connection_released():
    events = []
    class Connection:
        @contextmanager
        def transaction(self):
            events.append("begin")
            yield
            events.append("commit")
        def execute(self, sql, args=None):
            events.append((sql, args))
            result = Mock()
            result.fetchone.return_value = {"rolsuper": False, "rolbypassrls": False, "unsafe": False}
            result.fetchall.return_value = [chunk().model_dump()]
            return result
    class Pool:
        @contextmanager
        def connection(self):
            yield Connection()
            events.append("released")
    retriever = Retriever(settings())
    retriever.pool = Pool()
    assert retriever.search(7, "тест", np.ones(1024), 5)[0].chunk_id == 2
    assert events[0] == "begin"
    assert events[-2:] == ["commit", "released"]
    assert ("SELECT set_config('app.user_id', %s, true)", ("7",)) in events


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_context_overflow_is_actionable_without_exposing_server_prompt(stream,nested):
    message="private source text: prompt exceeds context length (n_keep: 4987 >= n_ctx: 4096)"
    error={"message":message} if nested else message
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            400,json={"error":error}))) as http:
        generator=Generator(settings(llm_provider="lmstudio"),http)
        with pytest.raises(ServiceError) as caught:
            if stream:list(generator.stream_answer("вопрос",[chunk()]))
            else:generator.answer("вопрос",[chunk()])
        assert "LLM context limit exceeded" in str(caught.value)
        assert "DEEP_CONTEXT_CHARS" in str(caught.value)
        assert "private source" not in str(caught.value)
        assert caught.value.status_code==502
