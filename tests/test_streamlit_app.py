from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import app
import app.components
import app.i18n
import app.runtime as app_runtime
import app.styles
import recommender.pipeline as pipeline_module
from app.i18n import TEXT
from recommender.config import load_config, load_prompts, project_root
from recommender.llm import OllamaLocalLLMClient
from recommender.schemas import (
    GenerationAudit,
    PipelineTrace,
    QueryParseAudit,
    UserFacingResult,
)


def test_i18n_has_matching_zh_and_en_keys():
    assert set(TEXT["zh"]) == set(TEXT["en"])


def test_streamlit_entrypoint_does_not_shadow_app_package():
    root = Path(__file__).resolve().parents[1]
    assert (root / "streamlit_app.py").is_file()
    assert not (root / "app" / "app.py").exists()
    assert app.__file__
    assert app.components.__package__ == "app"
    assert app.i18n.__package__ == "app"
    assert app.runtime.__package__ == "app"
    assert app.styles.__package__ == "app"


def test_cached_ollama_client_has_no_query_specific_mutable_state():
    client = OllamaLocalLLMClient(load_config().llm)
    try:
        keys = set(vars(client))
        assert not keys & {
            "query", "raw_query", "prompt", "response", "last_response",
            "last_query", "last_prompt", "last_result",
        }
        assert keys == {"config", "model", "_http", "_model_validated", "_model_validation_lock"}
    finally:
        client.close()


def test_streamlit_initial_render_is_offline_and_does_not_run_inference():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    assert not app.exception
    assert len(app.tabs) == 4
    assert app.session_state["last_result"] is None
    assert app.session_state["last_query"] == ""
    assert app.session_state["query_input"] == ""


def test_streamlit_submit_runs_once_and_plain_rerun_does_not_repeat(monkeypatch):
    config = load_config()
    prompts = load_prompts(project_root() / config.llm.prompt_path)
    assets = SimpleNamespace(
        config=config,
        prompts=prompts,
        resources=[],
        provenance_records=[],
        evidence_corpus=None,
        query_client=object(),
        final_client=object(),
    )
    calls: list[str] = []

    def fake_pipeline(query: str, **kwargs):
        calls.append(query)
        query_audit = QueryParseAudit(
            backend="mock", model="mock", prompt_version="summer_query_v2",
            parse_status="ERROR", retry_count=0, latency_ms=1,
            query_hash="sha256:" + "a" * 64, query_length=len(query),
        )
        generation = GenerationAudit(
            backend="mock", model="mock", prompt_version="summer_final_v1",
            generation_status="NOT_RUN", retry_count=0, latency_ms=0,
            grounding_valid=False, fallback_used=False, resource_count=0,
            input_character_count=0,
        )
        return UserFacingResult(
            status="QUERY_ERROR",
            query_audit=query_audit,
            generation_audit=generation,
            pipeline_trace=PipelineTrace(
                query_status="ERROR", query_latency_ms=1,
                recommendation_status="NOT_RUN", core_latency_ms=0,
                candidate_count=0, claim_count=0, retrieved_evidence_count=0,
                supported_count=0, unknown_count=0, contradicted_count=0,
                accepted_count=0, abstained_count=0,
                generation_status="NOT_RUN", generation_latency_ms=0,
                total_latency_ms=1,
            ),
            error_code="QUERY_PARSE_FAILED",
            error_message="test",
        )

    monkeypatch.setattr(app_runtime, "load_runtime_assets", lambda: assets)
    monkeypatch.setattr(pipeline_module, "run_user_pipeline", fake_pipeline)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    app.text_area[0].input("test query")
    app.button[0].click().run(timeout=15)
    assert calls == ["test query"]
    app.run(timeout=15)
    assert calls == ["test query"]


def test_streamlit_clear_resets_visible_query_and_result_without_inference(monkeypatch):
    config = load_config()
    prompts = load_prompts(project_root() / config.llm.prompt_path)
    assets = SimpleNamespace(
        config=config,
        prompts=prompts,
        resources=[],
        provenance_records=[],
        evidence_corpus=None,
        query_client=object(),
        final_client=object(),
    )
    calls: list[str] = []

    def fake_pipeline(query: str, **kwargs):
        calls.append(query)
        query_audit = QueryParseAudit(
            backend="mock", model="mock", prompt_version="summer_query_v2",
            parse_status="ERROR", retry_count=0, latency_ms=1,
            query_hash="sha256:" + "b" * 64, query_length=len(query),
        )
        generation = GenerationAudit(
            backend="mock", model="mock", prompt_version="summer_final_v1",
            generation_status="NOT_RUN", retry_count=0, latency_ms=0,
            grounding_valid=False, fallback_used=False, resource_count=0,
            input_character_count=0,
        )
        return UserFacingResult(
            status="QUERY_ERROR",
            query_audit=query_audit,
            generation_audit=generation,
            pipeline_trace=PipelineTrace(
                query_status="ERROR", query_latency_ms=1,
                recommendation_status="NOT_RUN", core_latency_ms=0,
                candidate_count=0, claim_count=0, retrieved_evidence_count=0,
                supported_count=0, unknown_count=0, contradicted_count=0,
                accepted_count=0, abstained_count=0,
                generation_status="NOT_RUN", generation_latency_ms=0,
                total_latency_ms=1,
            ),
            error_code="QUERY_PARSE_FAILED",
            error_message="test",
        )

    monkeypatch.setattr(app_runtime, "load_runtime_assets", lambda: assets)
    monkeypatch.setattr(pipeline_module, "run_user_pipeline", fake_pipeline)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    streamlit = AppTest.from_file(str(app_path)).run(timeout=15)
    streamlit.text_area[0].input("query to clear")
    streamlit.button[0].click().run(timeout=15)
    assert calls == ["query to clear"]
    assert streamlit.session_state["last_result"] is not None

    streamlit.button[1].click().run(timeout=15)
    assert calls == ["query to clear"]
    assert streamlit.text_area[0].value == ""
    assert streamlit.session_state["query_input"] == ""
    assert streamlit.session_state["last_query"] == ""
    assert streamlit.session_state["last_result"] is None

    streamlit.run(timeout=15)
    assert calls == ["query to clear"]
    assert streamlit.text_area[0].value == ""
