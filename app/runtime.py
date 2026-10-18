"""Cached immutable datasets and stateless/cached local transports for Streamlit."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from recommender.config import (
    PromptsConfig,
    SummerConfig,
    load_config,
    load_prompts,
    project_root,
    resolve_project_path,
)
from recommender.data import (
    EvidenceCorpus,
    load_evidence_corpus,
    load_resource_provenance,
    load_resources,
)
from recommender.llm import OllamaLocalLLMClient


@dataclass(frozen=True)
class RuntimeAssets:
    config: SummerConfig
    prompts: PromptsConfig
    resources: list
    provenance_records: list[dict]
    evidence_corpus: EvidenceCorpus
    query_client: OllamaLocalLLMClient
    final_client: OllamaLocalLLMClient


@st.cache_resource(show_spinner=False)
def load_runtime_assets() -> RuntimeAssets:
    root = project_root()
    config = load_config()
    prompts = load_prompts(resolve_project_path(config.llm.prompt_path, root=root))
    resources = load_resources(resolve_project_path(config.paths.resources, root=root))
    provenance = load_resource_provenance(
        resolve_project_path(config.paths.resource_provenance, root=root)
    )
    evidence = load_evidence_corpus(
        documents_path=resolve_project_path(config.paths.evidence_documents, root=root),
        chunks_path=resolve_project_path(config.paths.evidence_chunks, root=root),
        evidence_manifest_path=resolve_project_path(config.paths.evidence_manifest, root=root),
        resource_manifest_path=resolve_project_path(config.paths.resource_manifest, root=root),
        resources=resources,
        compatibility=config.evidence_compatibility,
    )
    # Clients retain only the HTTP pool and model-validation cache. Responses,
    # prompts, and query-specific state are returned to callers, never stored here.
    query_client = OllamaLocalLLMClient(config.llm)
    final_client = OllamaLocalLLMClient(config.llm.model_copy(update={
        "timeout_seconds": config.generation.timeout_seconds,
    }))
    return RuntimeAssets(
        config=config,
        prompts=prompts,
        resources=resources,
        provenance_records=provenance,
        evidence_corpus=evidence,
        query_client=query_client,
        final_client=final_client,
    )


__all__ = ["RuntimeAssets", "load_runtime_assets"]
