# S3 Grounded Final Generation Benchmark

## Status

- Evaluation date: 2026-09-15
- Branch: `feature/final-generation-ui`
- Baseline commit: `d71bcdb9a309a40332be5f8811efceb3299b6c82`
- Summer stage: S3 Grounded Final Generation and Streamlit UI
- Final prompt version: `summer_final_v1`
- Benchmark scale: 20 fixed end-to-end queries
- Result: passed the automated grounding/runtime baseline and local user acceptance

This benchmark evaluates the S3 explanation layer after real S2 Query
Understanding and the deterministic S1 trustworthy core. It does not treat
language fluency as evidence correctness, and it does not evaluate research-scale
recommendation quality.

## Method

The benchmark reuses the 20 representative queries in
`tests/fixtures/query_understanding/benchmark.jsonl`. One persistent local Ollama
client parses all queries and a second persistent local Ollama client performs
Final Generation. Successful model-tag validation is cached once per client;
neither client stores query-specific responses, prompts, or mutable user state.

Each query runs through:

```text
Raw Query
  -> qwen3:8b StructuredIntent
  -> S1 trustworthy core
  -> GroundedResultPack
  -> bounded generation view
  -> qwen3:8b summary/reasons
  -> grounding validation
  -> backend fact hydration
```

The Final LLM returns only a summary and one reason keyed by each backend-selected
`resource_id`. It cannot return fact text, Claim IDs, Evidence IDs, ranks,
scores, verification labels, `generation_method`, or the confidence disclaimer.
Those fields are resolved from canonical backend objects after validation.

The benchmark automatically checks:

- structured Final JSON validity;
- exact backend resource-set preservation;
- ranking preservation after backend hydration;
- Claim and Evidence references against the real S1 result;
- hallucinated Resource, Claim, and Evidence-ID counts;
- fallback count and actual fallback cause;
- Query Understanding, S1 core, Final Generation, and total latency.

The single query that S1 rejected did not invoke Final Generation and is reported
as `RECOMMENDATION_ABSTAINED`, not as a generation failure. Rates for Final JSON
and grounding therefore use the 19 genuinely attempted generations as their
denominator.

## Deterministic Input Budget

The canonical Evidence Corpus remains unchanged and the UI can still show all
retrieved Evidence. Only the view sent to Final Qwen is bounded:

| Limit | Value |
|---|---:|
| Maximum claims per resource | 8 |
| Maximum Evidence excerpts per Claim | 2 |
| Maximum Claim text characters | 280 |
| Maximum Evidence excerpt characters | 320 |
| Maximum serialized generation-view characters | 7,500 |
| Model context length | 4,096 tokens |
| Maximum generated tokens | 768 |

Truncation affects excerpt text only. Resource, Claim, Evidence, omitted-Claim,
and omitted-Evidence provenance IDs remain available in the bounded view or its
explicit omission lists. The full `GroundedResultPack` is retained for backend
hydration and UI inspection.

An initial 12,000-character trial reached 4,017 prompt tokens in the most extreme
case, leaving insufficient context headroom. After lowering the deterministic
view cap to 7,500 characters and output cap to 768 tokens, the full benchmark's
maximum was 2,676 prompt tokens and 306 output tokens.

## Runtime Environment

| Component | Value |
|---|---|
| Model | `qwen3:8b` |
| Ollama | `0.34.0` |
| Python | `3.11.16` in the project Conda environment |
| Operating system | Windows x64, build 26200 |
| CPU | Intel Core i7-14650HX |
| System memory | 32 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM |
| GPU driver | `616.92` |
| Canonical data | 20 Resources / 40 Evidence Documents / 680 Evidence Chunks |
| Offline test suite | 123 passed |

Latency is specific to this machine, model state, and system load. It is not a
portable performance claim.

## Results

| Metric | Result |
|---|---:|
| Query count | 20 |
| S1 accepted / Final Generation attempted | 19 |
| S1 recommendation abstentions | 1 |
| Final JSON Valid Rate | 19/19 (100%) |
| Grounding Pass Rate | 19/19 (100%) |
| Hallucinated Resource Count | 0 |
| Hallucinated Claim Count | 0 |
| Hallucinated Evidence ID Count | 0 |
| Ranking Mutation Count | 0 |
| Fallback Count | 0 |
| Average Query Understanding latency | 3,392.840 ms |
| Average S1 core latency | 25.674 ms |
| Average Final Generation latency | 6,627.939 ms |
| Final Generation nearest-rank P95 | 9,268.651 ms |
| Average total pipeline latency | 9,715.401 ms |
| Total pipeline nearest-rank P95 | 12,107.854 ms |
| Maximum generation-view characters | 7,038 |
| Average / maximum prompt tokens | 1,881.368 / 2,676 |
| Average / maximum output tokens | 230.789 / 306 |

The grounding metrics check identity and provenance invariants. They do not claim
that every generated sentence is stylistically ideal. Output-language adherence
and readability remain auxiliary human-review dimensions; no language detector
is used as a factual grounding gate.

## Web Smoke

The Streamlit application was checked in two ways:

- offline `streamlit.testing.v1.AppTest` coverage for initial rendering, one
  submit action, and rerun safety;
- a real headless Streamlit server startup on localhost, which returned HTTP 200
  with the Streamlit page before the test server was stopped.
- local acceptance of the query form, clear action, Evidence/Verification,
  Confidence, Abstention, Pipeline Trace, and rerun behavior.

No browser action, network fetch, or canonical data mutation is required by the
web application. Source links are display-only and point to provenance already
stored in the canonical Evidence Corpus.

## Reproducibility

With Ollama running locally and `qwen3:8b` installed:

```powershell
python -m recommender.final_benchmark
```

The command reads canonical project data without modifying it and prints only the
aggregate summary by default. To retain per-query audit rows for local human
review, explicitly use an ignored path:

```powershell
python -m recommender.final_benchmark --output logs/summer/s3_benchmark_results.jsonl
```

Generated logs remain excluded from Git. Reruns may vary in prose and latency,
so they should record their own runtime metrics instead of assuming these values.

## Known Limitations

- Twenty hand-authored queries and 20 Resources form a validation-scale test, not
  a research-scale recommendation evaluation.
- Resources are currently GitHub repositories, and Evidence is mainly README plus
  repository metadata from the canonical Phase 3 corpus.
- Candidate Retrieval is TF-IDF, the KG is simple, and Verification is rule-based.
- Confidence is heuristic and uncalibrated; it is not a probability.
- Human-language preference is not fully represented in the Resource Dataset.
- Grounding validation constrains identities and backend facts, but local Qwen
  prose remains model-generated and is not assumed infallible.
- Language quality has not been assigned a formal automatic score. It requires
  human review and is intentionally separate from factual grounding.
- No Embedding retrieval, Vector DB, learned verifier, calibrated uncertainty,
  cloud LLM, model fine-tuning, or production security/performance hardening is
  included in S3.
