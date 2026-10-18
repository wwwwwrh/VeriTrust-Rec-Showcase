# S2 Query Understanding Benchmark

## Status

- Evaluation date: 2026-09-15
- Summer stage: S2 Local Query Understanding
- Prompt version: `summer_query_v2`
- Benchmark scale: 20 fixed queries
- Result: passed the current Summer Edition acceptance baseline

This document records the versioned evaluation of the local Query Understanding
layer. It does not evaluate S1 candidate retrieval, evidence verification,
confidence, ranking, or later S3–S4 components.

## Method

The benchmark uses the 20 cases in
`tests/fixtures/query_understanding/benchmark.jsonl`. The cases cover Chinese,
English, and mixed-language requests; explicit skills and prerequisites;
framework preferences and exclusions; difficulty; learning style; time budget;
human-language preference; vague requests; and one out-of-domain request.

Each case was sent to one persistent local Ollama client with:

- JSON-Schema-constrained `StructuredIntent` output
- `temperature: 0.0`
- `context_length: 4096`
- `max_tokens: 512`
- `thinking: false`
- no cloud fallback and no network service other than the loopback Ollama endpoint

Before the run, `qwen3:8b` was unloaded. The first request therefore measures
cold-start latency. The following 19 requests measure warm inference on the same
client. Warm P95 uses nearest-rank P95.

Schema validity and scalar `difficulty` equality are computed automatically.
The list-valued semantic fields are reviewed at field level rather than requiring
literal string equality. In particular, `NLP` and `Natural Language Processing`,
or `LLM` and `Large Language Models`, are accepted as semantically equivalent
canonical English technical terms.

Human review checks all nine fields for every query:

1. `topics`
2. `skills`
3. `prerequisites`
4. `framework_preferences`
5. `language`
6. `difficulty`
7. `learning_style`
8. `time_budget`
9. `constraints`

Unsupported inference is counted whenever the model introduces a preference,
ability, requirement, or topic not expressed by the user. Negative Constraint
Accuracy covers the three benchmark cases with explicit negative or exclusion
semantics (`q04`, `q07`, and `q14`).

The review also distinguishes soft preferences from hard constraints. Phrases
such as `最好`, `希望`, `prefer`, and `preferably` must not be strengthened into
`must` or `required`. Soft hands-on preferences belong in `learning_style` and
are not duplicated in `constraints`; only explicit hard wording such as `必须`,
`一定要`, `不能`, `不要`, `must`, `required`, or `exclude` creates a hard
constraint.

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

Latency values are specific to this machine and runtime state. They are not a
portable performance claim.

## Results

| Metric | Result |
|---|---:|
| Query count | 20 |
| Schema Valid Rate | 20/20 (100%) |
| Difficulty Exact Match | 20/20 (100%) |
| Human field-level semantic correctness | 180/180 (100%) |
| Explicit canonical-English mapping checks | 9/9 (100%) |
| Unsupported Inference Count | 0 |
| Negative Constraint Accuracy | 3/3 (100%) |
| Dedicated-field duplication in `constraints` | 0 |
| Soft preferences strengthened to hard constraints | 0 |
| Query-processing errors | 0 |
| Cold-start latency | 6,400.272 ms |
| Warm average latency | 1,552.081 ms |
| Warm nearest-rank P95 latency | 1,875.120 ms |

The theory regression case now produces `learning_style=["theory"]`. The explicit
Python prerequisite case places `Python basics` only in `prerequisites` and does
not duplicate it in `constraints`. All explicitly mapped Chinese AI/ML terms in
the benchmark use canonical English terminology.

The soft-preference regression case produces `learning_style=["hands_on"]` with
`constraints=[]`. The benchmark's mixed-language hands-on case does the same.
The hard wording `不要只有理论` remains auditable as
`constraints=["exclude theory-only resources"]` without inventing a mandatory
code-example requirement.

Sixteen predictions were literal full-object matches to their fixture examples.
The other four differed only by expanding the user-written acronyms `NLP` or
`LLM` to `Natural Language Processing` or `Large Language Models`; these were
counted as semantically correct under the approved field-level methodology.

## Reproducibility

With Ollama already running locally and `qwen3:8b` installed, run:

```powershell
ollama stop qwen3:8b
python -m recommender.query_benchmark --output logs/summer/s2_benchmark_results.jsonl
```

The raw prediction log is intentionally written under `logs/` and is excluded
from Git. This report is the reviewable, versioned summary. Because local model
generation and machine load can vary, reruns must record their own latency and
semantic review rather than assuming these numbers.

## Known Limitations

- Twenty hand-authored queries are a validation-scale benchmark, not a broad
  natural-language-understanding evaluation dataset.
- Manual semantic review has reviewer judgment and does not replace independent
  annotation or inter-annotator agreement.
- The benchmark emphasizes Chinese and English AI/ML learning requests and has
  limited linguistic, domain, and adversarial coverage.
- Canonical terminology is prompt-guided rather than enforced by a controlled
  vocabulary layer, so surface forms may vary between semantically equivalent
  abbreviations and expanded names.
- Temperature zero reduces variation but does not guarantee byte-identical model
  output across Ollama/model/runtime versions.
- Latency is hardware- and runtime-specific; cold start includes model loading.
- `StructuredIntent.language` remains a human-language preference and is not used
  as a programming-language retrieval signal in the current Summer Edition.
- This benchmark does not validate recommendation quality, calibrated confidence,
  final generation, Streamlit, or S3–S4 evaluation behavior.
