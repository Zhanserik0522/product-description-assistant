import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.llm import ModelTimeout, ModelUnavailable, OllamaClient
from app.main import app, get_llm


@pytest.fixture
def product():
    return {
        "name": "Манометр DM 02", "manufacturer": "Манотомь",
        "specs": {"диапазон": "0–1,6 МПа", "питание": None},
        "description": "Измерение избыточного давления.",
    }


@pytest.fixture
def draft():
    # Тестовый ответ заглушки: здесь проверяются API и валидация, не качество LLM.
    return {
        "seo_title": "Манометр DM 02",
        "seo_description": "Прибор для измерения избыточного давления. " * 22,
        "search_queries": ["манометр DM 02", "Манотомь DM 02", "манометр давления"],
        "application_quote": "Измерение избыточного давления.",
    }


class FakeLLM:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = 0

    async def generate(self, messages):
        self.calls += 1
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer, ensure_ascii=False) if isinstance(answer, dict) else answer


@pytest.fixture
def api():
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def use_model(model):
    app.dependency_overrides[get_llm] = lambda: model


def test_card_preserves_source_and_marks_missing(api, product, draft):
    use_model(FakeLLM(draft))
    response = api.post("/generate", json=product)
    assert response.status_code == 200
    data = response.json()
    assert data["characteristics"] == {"диапазон": "0–1,6 МПа", "питание": "данных нет"}
    assert data["missing_fields"] == ["specs.питание"]
    assert 800 <= len(data["seo_description"]) <= 1200
    assert product["description"] in data["commercial_block"]


@pytest.mark.parametrize("payload", [
    {}, {"name": "   "}, {"name": "X", "specs": []},
    {"name": "X", "specs": {"": "Y"}},
    {"name": "X", "specs": {"a": "x", " a ": "y"}},
    {"name": "X", "unexpected": "value"},
])
def test_invalid_input(api, payload):
    assert api.post("/generate", json=payload).status_code == 422


def test_malformed_json(api):
    response = api.post("/generate", content='{ "name": ',
                        headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_empty_optional_fields(api, draft):
    draft["application_quote"] = None
    use_model(FakeLLM(draft))
    data = api.post("/generate", json={"name": "Манометр DM 02",
                    "manufacturer": "  ", "description": ""}).json()
    assert set(data["missing_fields"]) == {"manufacturer", "description", "specs"}
    assert data["characteristics"] == {"Характеристики": "данных нет"}


def test_retry_invalid_json(api, product, draft):
    model = FakeLLM("not json", draft)
    use_model(model)
    assert api.post("/generate", json=product).status_code == 200
    assert model.calls == 2


def test_decimal_separator_is_not_a_new_number(api, product, draft):
    draft["search_queries"][0] = "манометр 0–1.6 МПа"
    use_model(FakeLLM(draft))
    assert api.post("/generate", json=product).status_code == 200


@pytest.mark.parametrize("change", [
    {"seo_title": "x" * 71},
    {"seo_description": "Слишком коротко"},
    {"seo_description": "x" * 1201},
    {"search_queries": ["одинаково", "одинаково", "одинаково"]},
    {"seo_title": "Питание 220 В"},
    {"seo_title": "Манометр по ГОСТу"},
    {"application_quote": "Для газов и жидкостей"},
])
def test_bad_output_is_not_returned(api, product, draft, change):
    invalid = draft | change
    model = FakeLLM(invalid, invalid)
    use_model(model)
    assert api.post("/generate", json=product).status_code == 502
    assert model.calls == 2


@pytest.mark.parametrize("error, status", [(ModelTimeout(), 504), (ModelUnavailable(), 503)])
def test_model_errors(api, product, error, status):
    use_model(FakeLLM(error))
    assert api.post("/generate", json=product).status_code == status


def test_ollama_transport_and_schema(monkeypatch, draft):
    import asyncio

    async def handle(request):
        body = json.loads(request.content)
        assert body["format"]["type"] == "object"
        assert body["stream"] is False
        assert body["think"] is False
        return httpx.Response(200, json={"message": {"content": json.dumps(draft)}})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw:
                        original(transport=httpx.MockTransport(handle), **kw))
    result = asyncio.run(OllamaClient().generate([]))
    assert json.loads(result) == draft


@pytest.mark.parametrize("failure, expected", [
    ("timeout", ModelTimeout), ("status", ModelUnavailable),
    ("bad_json", ModelUnavailable), ("missing_content", ModelUnavailable),
])
def test_ollama_failure_mapping(monkeypatch, failure, expected):
    import asyncio

    async def handle(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "status":
            return httpx.Response(500)
        if failure == "bad_json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, json={})

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw:
                        original(transport=httpx.MockTransport(handle), **kw))
    with pytest.raises(expected):
        asyncio.run(OllamaClient().generate([]))
