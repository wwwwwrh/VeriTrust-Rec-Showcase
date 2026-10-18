from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

import recommender.pipeline as pipeline_module
from recommender.config import LLMConfig, load_config, load_prompts, project_root
from recommender.errors import QueryErrorCode, QueryProcessingError
from recommender.llm import MockLLMClient, OllamaLocalLLMClient, parse_query
from recommender.llm.query_parser import structured_intent_response_schema
from recommender.retrieval.retriever import intent_to_query
from recommender.schemas import (
    QueryParseAudit,
    StructuredIntent,
    SummerPipelineResult,
    SummerQueryResult,
)


VALID_INTENT = {
    "topics": ["Transformer"],
    "skills": [],
    "prerequisites": [],
    "framework_preferences": [],
    "language": [],
    "difficulty": "unknown",
    "learning_style": [],
    "time_budget": None,
    "constraints": [],
}


@pytest.fixture
def settings():
    config = load_config()
    config = config.model_copy(update={
        "logging": config.logging.model_copy(update={"save_sessions": False})
    })
    prompts = load_prompts(project_root() / config.llm.prompt_path)
    return config, prompts


def parse_with_mock(responses: list[str | BaseException], settings):
    config, prompts = settings
    client = MockLLMClient(responses)
    return parse_query(
        "我想学习 Transformer。",
        client,
        llm_config=config.llm,
        prompt_config=prompts.query_understanding,
    ), client


def test_empty_query_is_a_query_error_without_calling_model(settings):
    config, prompts = settings
    client = MockLLMClient([])
    with pytest.raises(QueryProcessingError) as caught:
        parse_query(
            "   ", client,
            llm_config=config.llm,
            prompt_config=prompts.query_understanding,
        )
    assert caught.value.code == QueryErrorCode.QUERY_EMPTY
    assert client.calls == []


def test_valid_response_is_strict_structured_intent(settings):
    parsed, client = parse_with_mock([json.dumps(VALID_INTENT)], settings)
    assert parsed.intent == StructuredIntent(topics=["Transformer"])
    assert parsed.audit.parse_status == "SUCCESS"
    assert parsed.audit.retry_count == 0
    call = client.calls[0]
    assert call["thinking"] is False
    assert set(call["schema"]["required"]) == set(VALID_INTENT)
    assert call["schema"]["additionalProperties"] is False
    assert not ({"recommendations", "resources", "confidence"} & set(call["schema"]["properties"]))


def test_summer_query_v2_prompt_enforces_lexical_normalization_and_field_separation(settings):
    _, prompts = settings
    prompt = prompts.query_understanding.system_prompt
    flat_prompt = " ".join(prompt.split())
    assert prompts.query_understanding.prompt_version == "summer_query_v2"
    assert "Computer Vision" in flat_prompt
    assert "Natural Language Processing" in flat_prompt
    assert "lexical normalization only" in flat_prompt
    assert "MUST use the listed English term" in flat_prompt
    assert "learning_style MUST" in flat_prompt
    assert "include theory" in flat_prompt
    assert "MUST NOT be strengthened into must" in flat_prompt
    assert "do not repeat it in constraints" in flat_prompt
    assert "A time budget must not change difficulty" in flat_prompt
    assert "Preserve an explicitly requested out-of-domain topic" in flat_prompt


def test_soft_code_preference_contract_is_not_strengthened(settings):
    _, prompts = settings
    flat_prompt = " ".join(prompts.query_understanding.system_prompt.split())
    assert "最好有代码实践" in flat_prompt
    assert 'learning_style=["hands_on"] and constraints=[]' in flat_prompt
    assert 'Emit "must include code examples" only when code examples are explicitly mandatory' in flat_prompt


def test_benchmark_distinguishes_soft_code_preference_from_hard_exclusion():
    path = project_root() / "tests/fixtures/query_understanding/benchmark.jsonl"
    cases = {
        row["case_id"]: row
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }
    assert cases["q12_mixed"]["expected"]["learning_style"] == ["hands_on"]
    assert cases["q12_mixed"]["expected"]["constraints"] == []
    assert cases["q07_hands_on"]["expected"]["constraints"] == [
        "exclude theory-only resources"
    ]


def test_structured_schema_describes_v2_semantic_boundaries():
    properties = structured_intent_response_schema()["properties"]
    assert "canonical English" in properties["topics"]["description"]
    assert "requires theory" in properties["learning_style"]["description"]
    assert "never duplicate" in properties["constraints"]["description"]


@pytest.mark.parametrize(
    "bad_update",
    [
        {"invented": True},
        {"difficulty": "expert"},
    ],
)
def test_extra_fields_and_invalid_difficulty_retry_then_fail(bad_update, settings):
    invalid = VALID_INTENT | bad_update
    with pytest.raises(QueryProcessingError) as caught:
        parse_with_mock([json.dumps(invalid), json.dumps(invalid)], settings)
    assert caught.value.code == QueryErrorCode.QUERY_PARSE_FAILED
    assert caught.value.retry_count == 1


def test_retry_first_invalid_then_success(settings):
    parsed, client = parse_with_mock(["not json", json.dumps(VALID_INTENT)], settings)
    assert parsed.intent.topics == ["Transformer"]
    assert parsed.audit.retry_count == 1
    assert len(client.calls) == 2
    assert "previous response" in client.calls[1]["system_prompt"].casefold()


def test_unknown_time_budget_is_only_conservative_sentinel_normalization(settings):
    value = VALID_INTENT | {"time_budget": " unknown "}
    parsed, _ = parse_with_mock([json.dumps(value)], settings)
    assert parsed.intent.time_budget is None

    explicit = VALID_INTENT | {"time_budget": "two weeks"}
    parsed, _ = parse_with_mock([json.dumps(explicit)], settings)
    assert parsed.intent.time_budget == "two weeks"


def test_negative_constraint_mock_only_preserves_valid_model_output(settings):
    value = VALID_INTENT | {
        "framework_preferences": ["PyTorch"],
        "constraints": ["exclude TensorFlow"],
    }
    parsed, _ = parse_with_mock([json.dumps(value)], settings)
    assert parsed.intent.framework_preferences == ["PyTorch"]
    assert parsed.intent.constraints == ["exclude TensorFlow"]


@pytest.mark.parametrize(
    ("code", "expected_calls"),
    [
        (QueryErrorCode.LOCAL_LLM_TIMEOUT, 1),
        (QueryErrorCode.LOCAL_LLM_UNAVAILABLE, 1),
    ],
)
def test_transport_errors_do_not_retry(code, expected_calls, settings):
    config, prompts = settings
    client = MockLLMClient([QueryProcessingError(code, "failure")])
    with pytest.raises(QueryProcessingError) as caught:
        parse_query(
            "query", client,
            llm_config=config.llm,
            prompt_config=prompts.query_understanding,
        )
    assert caught.value.code == code
    assert len(client.calls) == expected_calls


def test_invalid_transport_response_retries_once(settings):
    config, prompts = settings
    client = MockLLMClient([
        QueryProcessingError(QueryErrorCode.LOCAL_LLM_RESPONSE_INVALID, "bad envelope"),
        json.dumps(VALID_INTENT),
    ])
    parsed = parse_query(
        "query", client,
        llm_config=config.llm,
        prompt_config=prompts.query_understanding,
    )
    assert parsed.audit.retry_count == 1


def test_human_language_is_not_a_programming_language_retrieval_signal():
    baseline = StructuredIntent(topics=["Transformer"])
    chinese = StructuredIntent(topics=["Transformer"], language=["Chinese"])
    assert intent_to_query(baseline) == intent_to_query(chinese)


def test_language_only_intent_has_no_tfidf_signal():
    assert intent_to_query(StructuredIntent(language=["Chinese"])) == ""


def test_benchmark_has_twenty_schema_complete_cases():
    path = project_root() / "tests/fixtures/query_understanding/benchmark.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(cases) >= 20
    required = set(structured_intent_response_schema()["required"])
    assert all(set(case["expected"]) == required for case in cases)


def test_llm_config_rejects_non_loopback_and_thinking():
    with pytest.raises(ValidationError):
        LLMConfig(base_url="https://example.com")
    with pytest.raises(ValidationError):
        LLMConfig(thinking=True)


def test_ollama_payload_and_model_validation_are_cached():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={
                "models": [{"name": "qwen3:8b", "model": "qwen3:8b", "size": 123}]
            })
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["format"] == structured_intent_response_schema()
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": json.dumps(VALID_INTENT)}
        })

    client = OllamaLocalLLMClient(
        LLMConfig(), transport=httpx.MockTransport(handler)
    )
    try:
        for _ in range(2):
            client.generate_structured(
                system_prompt="system",
                user_prompt="query",
                schema=structured_intent_response_schema(),
                thinking=False,
            )
    finally:
        client.close()
    assert [request.url.path for request in requests].count("/api/tags") == 1
    assert [request.url.path for request in requests].count("/api/chat") == 2
    assert all("authorization" not in request.headers for request in requests)


def test_default_ollama_client_is_reused_across_queries(monkeypatch, settings):
    config, _ = settings
    created = []

    class FakeClient:
        def __init__(self, llm_config):
            self.config = llm_config
            created.append(self)

    monkeypatch.setattr(pipeline_module, "OllamaLocalLLMClient", FakeClient)
    pipeline_module._DEFAULT_LLM_CLIENTS.clear()
    first = pipeline_module._default_llm_client(config)
    second = pipeline_module._default_llm_client(config)
    assert first is second
    assert len(created) == 1
    pipeline_module._DEFAULT_LLM_CLIENTS.clear()


def test_missing_local_model_is_not_configured():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"models": []}))
    client = OllamaLocalLLMClient(LLMConfig(), transport=transport)
    try:
        with pytest.raises(QueryProcessingError) as caught:
            client.generate("query")
    finally:
        client.close()
    assert caught.value.code == QueryErrorCode.LOCAL_LLM_NOT_CONFIGURED


def test_ollama_timeout_and_unavailable_are_typed():
    for failure, expected in (
        (httpx.ReadTimeout("slow"), QueryErrorCode.LOCAL_LLM_TIMEOUT),
        (httpx.ConnectError("down"), QueryErrorCode.LOCAL_LLM_UNAVAILABLE),
    ):
        def handler(request: httpx.Request, error=failure) -> httpx.Response:
            raise error

        client = OllamaLocalLLMClient(
            LLMConfig(), transport=httpx.MockTransport(handler)
        )
        try:
            with pytest.raises(QueryProcessingError) as caught:
                client.generate("query")
        finally:
            client.close()
        assert caught.value.code == expected


def test_query_failure_does_not_execute_s1(monkeypatch, settings):
    config, prompts = settings
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("S1 must not run")

    monkeypatch.setattr(pipeline_module, "run_core_pipeline", forbidden)
    result = pipeline_module.recommend_from_query(
        "query",
        client=MockLLMClient(["bad", "still bad"]),
        config=config,
        prompts=prompts,
    )
    assert result.status == "QUERY_ERROR"
    assert result.error_code == QueryErrorCode.QUERY_PARSE_FAILED
    assert result.recommendation is None
    assert not called


def test_query_result_schema_rejects_failure_disguised_as_abstention():
    audit = QueryParseAudit(
        backend="mock",
        model="mock",
        prompt_version="summer_query_v2",
        parse_status="ERROR",
        retry_count=0,
        latency_ms=1,
        query_hash="sha256:" + "0" * 64,
        query_length=1,
    )
    with pytest.raises(ValidationError):
        SummerQueryResult(
            status="RECOMMENDATION_ABSTAINED",
            query_audit=audit,
            recommendation=SummerPipelineResult(status="ABSTAINED"),
        )


def test_recommendation_abstention_is_distinct_from_query_error(monkeypatch, settings):
    config, prompts = settings

    def abstained(intent, **kwargs):
        assert intent.topics == ["Transformer"]
        return SummerPipelineResult(
            status="ABSTAINED",
            intent=intent,
            error_code="NO_CANDIDATE",
            error_message="no candidate",
        )

    monkeypatch.setattr(pipeline_module, "run_core_pipeline", abstained)
    result = pipeline_module.recommend_from_query(
        "query",
        client=MockLLMClient([json.dumps(VALID_INTENT)]),
        config=config,
        prompts=prompts,
    )
    assert result.status == "RECOMMENDATION_ABSTAINED"
    assert result.parsed_query is not None
    assert result.recommendation is not None
    assert result.recommendation.status == "ABSTAINED"
    assert result.error_code is None


def test_query_audit_does_not_store_raw_query(settings):
    secret_like_query = "private learning request"
    config, prompts = settings
    parsed = parse_query(
        secret_like_query,
        MockLLMClient([json.dumps(VALID_INTENT)]),
        llm_config=config.llm,
        prompt_config=prompts.query_understanding,
    )
    payload = parsed.audit.model_dump_json()
    assert secret_like_query not in payload
    assert parsed.audit.query_length == len(secret_like_query)
