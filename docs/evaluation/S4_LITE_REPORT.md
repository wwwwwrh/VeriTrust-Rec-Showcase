# VeriTrust-Rec Summer Edition — S4-Lite Evaluation Report

> Historical report from the original development repository. Results and frozen identities below are unchanged. This independent snapshot omits the frozen corpus and quoted Verification Gold pending distribution-rights clearance, and does not inherit original Git history/tags. Commands below describe the original experiment, not an executable fresh-clone benchmark in this snapshot. Software and synthetic engineering tests remain available; they do not reproduce or replace the reported accuracy. See [benchmark availability](../../evaluation/benchmarks/README.md).

## 1. Scope and time-box

S4-Lite is a small, presentation-oriented evaluation of the accepted Summer Edition pipeline. It measures the existing TF-IDF retrieval, rule-based verification, trustworthy abstention, and grounded local generation. It does not tune or modify S1–S3 algorithms, thresholds, prompts, or the frozen dataset.

The benchmark contains 10 human-reviewed Retrieval cases, 15 blind-annotated Verification cases, three Abstention cases, one cold runtime run, five warm runtime runs, and two explainability examples. These results are preliminary course-project evidence, not a large-scale research evaluation.

## 2. Frozen dataset identity

| Field | Frozen value |
|---|---|
| Dataset tag | `summer-s3.5-resource-200` |
| Tag target / base commit | `693ab9b63fcccbf351a2590583d3e934833a3c18` |
| Raw run ID | `a2aa04a6225e406faeb28cfcfe5d6c29` |
| Curation manifest | `summer_s35_stage_c_final_v1` / `1.0.0` |
| Curation hash | `sha256:0346f405c22dc339e0d3abfc043ae05df6a4373b43969bae11428c2def9ac9cb` |
| Resource Build | `build_60c4f8a08dad8614` |
| Evidence Build | `evidence_build_10deed019a096c1a` |
| Corpus size | 200 Resources / 400 Documents / 5,902 Chunks |
| Evaluation input | `data/s35/stages/stage-c-final/data` |

The explicitly supplied Stage C final copy passed the complete identity check and has the same deterministic Resource/Evidence identities as the promoted canonical corpus. There was no implicit data-path fallback. All five curated exclusions were absent; orphan Document and Chunk counts were zero.

## 3. Evaluation setup and reproducibility

| Field | Value |
|---|---|
| Benchmark version/status | `1.0.0` / `human_confirmed` |
| Evaluator version | `s4_lite_evaluator_v1.0.0` |
| Benchmark hash | `sha256:87755e7d0fe4f46213b8680b38a9b79cda75900a69f03aa97fd35b5b40b552e3` |
| Evaluation config hash | `sha256:c1d7697282e0bde99562e815ec249e5150934cc554563cdf079d71a6526d2bbc` |
| Evaluation code hash | `sha256:f09d160375c2d170578c77ed312178b7ee2f1ca7538e894db062b34fe0604a7a` |
| Git state during evaluation | base commit above; `git_dirty=true` |
| Reference time | `2026-09-16T00:00:00Z` |
| Query prompt | `summer_query_v2` |
| Final prompt | `summer_final_v1` |
| Local model/runtime | Ollama `0.34.0`, `qwen3:8b` |
| Hardware | Intel Core i7-14650HX; NVIDIA RTX 4060 Laptop GPU (8,188 MiB); about 31.7 GiB RAM |

Because the evaluator was intentionally uncommitted during acceptance, the Git commit identifies only the base repository state. The evaluation code hash identifies the complete evaluator source used for this run.

Gold validation passed before metrics were run: all StructuredIntent objects were schema-valid; all acceptable Resource IDs existed; all Claim IDs recomputed to their canonical identities; all Resource, Evidence, and Chunk IDs resolved; Evidence text exactly matched the frozen corpus; same-resource binding held; and all labels belonged to `SUPPORTED`, `UNKNOWN`, or `CONTRADICTED`.

## 4. Retrieval results

| Metric | Result |
|---|---:|
| Cases | 10 |
| Top-1 Reasonable Rate | 6/10 = **60%** |
| Top-3 Hit Rate | 8/10 = **80%** |
| MRR@10 | **0.745** |

The Top-1 misses were PyTorch/Deep Learning, Computer Vision, Diffusion/Generative AI, and MLOps/deployment. The PyTorch and MLOps cases also had no acceptable result in the Top 3; their first acceptable results appeared at ranks 4 and 5 respectively. These failures show the limits of lexical matching when a broad framework/tool repository scores above a more educational resource.

## 5. Verification results

| Metric | Result |
|---|---:|
| Cases | 15 |
| Correct | 13 |
| Accuracy | **86.67%** |
| Macro-F1 | **0.8917** |

Gold labels were 9 SUPPORTED, 4 UNKNOWN, and 2 CONTRADICTED. Predictions were 7 SUPPORTED, 6 UNKNOWN, and 2 CONTRADICTED.

| Gold \ Predicted | SUPPORTED | UNKNOWN | CONTRADICTED |
|---|---:|---:|---:|
| SUPPORTED | 7 | 2 | 0 |
| UNKNOWN | 0 | 4 | 0 |
| CONTRADICTED | 0 | 0 | 2 |

The two errors were conservative false negatives:

- `verification_001`: Dive-into-DL-PyTorch / `uses_framework`. Gold was SUPPORTED from explicit README language, while the rule returned UNKNOWN.
- `verification_006`: model deployment / `covers_topic`. Gold was SUPPORTED from repository metadata and README language, while the rule returned UNKNOWN.

No UNKNOWN or CONTRADICTED Gold case was incorrectly promoted to SUPPORTED.

## 6. Abstention results

All 3/3 functional cases passed:

- A normal PyTorch request returned `OK` with recommendations.
- An out-of-domain Cooking request returned `ABSTAINED / NO_CANDIDATE`.
- A controlled test using only the real archived `tensorflow/tensor2tensor` Resource and its canonical Evidence returned `ABSTAINED / ALL_CANDIDATES_ABSTAINED`.

The controlled contradiction case is a functional real-corpus test, not an estimate of population-level contradiction frequency.

## 7. Runtime results

The local runtime protocol used one unloaded-model cold run followed by five warm runs on the same client instances. Session logging was disabled and the frozen data remained read-only.

| Stage | Cold (ms) | Warm average (ms) | Warm median (ms) |
|---|---:|---:|---:|
| Query Understanding | 8,950.536 | 3,795.629 | 3,782.609 |
| S1 core | 78.762 | 79.222 | 72.942 |
| Final Generation | 8,842.566 | 5,661.122 | 5,630.890 |
| Total | 17,872.494 | 9,536.552 | 9,537.472 |

All six measured runs returned `OK`, successful grounded generation, and identical ranked Resource IDs. Across the measured runs, ranking mutations and hallucinated Resource, Claim, or Evidence IDs were all zero.

The headless Streamlit server started on localhost, returned HTTP 200, and rendered the complete Query UI in a real browser session. An end-to-end query submitted from that Web session could not load the default canonical Resource path under the Codex sandbox identity. This is an execution-identity/ACL visibility limitation: the explicit frozen copy used by S4-Lite passed all pipeline checks, but the existing Streamlit entry point has no explicit `--data-root` override. No canonical data, ACL, or S1–S3 code was changed to bypass it.

## 8. Explainability examples

### PyTorch / Deep Learning

The pipeline returned four Resources in this order: `github:33015583`, `github:1198539`, `github:200645982`, and `github:171795571`. The final explanation preserved that order. Backend-owned facts included canonical maintenance, topic, framework, and license Claims with real Evidence Chunk IDs. UNKNOWN facts—such as whether Netron itself uses PyTorch—remained explicitly uncertain instead of being presented as verified.

### LLM / RAG

The pipeline returned `github:740303686`, `github:655099582`, `github:600544354`, `github:674075444`, and `github:47184304`. The generated explanation preserved the ranking and cited only canonical Claim/Evidence identities. RAG Claims without adequate support remained UNKNOWN, even where the resource was otherwise relevant to LLMs or Transformers.

In both examples, factual text was resolved by the backend from canonical Claims. Qwen generated only the summary and reasons. The displayed confidence remains a prototype heuristic score, not a calibrated probability of correctness.

## 9. Error analysis

Retrieval errors mainly reflect sparse Resource metadata and lexical ambiguity. A repository may contain the right framework term but be less suitable as a learning resource than a lower-ranked tutorial. The current retriever does not use semantic embeddings or a learned educational-quality signal.

Verification errors were both conservative SUPPORTED-to-UNKNOWN errors. This behavior reduces unsupported positive assertions, but it also misses some explicit natural-language support that falls outside narrow rule patterns. S4-Lite records the limitation without changing verifier rules after observing the test set.

## 10. Known limitations

- The corpus contains only GitHub repositories and is not representative of all learning-resource platforms.
- Ten Retrieval and fifteen Verification cases are too small for strong statistical conclusions.
- Acceptable Retrieval sets are human judgments from one annotation round.
- TF-IDF is lexical and cannot fully represent intent, pedagogy, or semantic equivalence.
- Verification uses conservative deterministic rules rather than NLI.
- Runtime was measured on one Windows laptop and one local model/runtime configuration.
- The source-reliability `heuristic_v0` values are non-calibrated priors, not trustworthiness scores.
- Confidence is heuristic and uncalibrated.

## 11. Reproduction commands

From the repository root in the `veritrust-rec` Conda environment:

```powershell
python -m evaluation.s4_lite preflight --data-root data/s35/stages/stage-c-final/data
python -m evaluation.s4_lite evaluate --data-root data/s35/stages/stage-c-final/data
python -m evaluation.s4_lite runtime --data-root data/s35/stages/stage-c-final/data --warm-runs 5 --output-language zh
python -m pytest -q --basetemp=pytest_s4_wrh -p no:cacheprovider
```

Runtime results require a local Ollama service with `qwen3:8b`. The JSON audit outputs under `evaluation/results/` are intentionally ignored by Git.

## 12. Relationship to a full S4 research evaluation

S4-Lite provides a compact, reproducible course-project evaluation and presentation baseline. A full research S4 would require a larger multi-annotator benchmark, agreement measurement, broader data sources, stronger retrieval baselines, calibrated uncertainty, statistical confidence intervals, failure stratification, and controlled model/hardware comparisons. None of those stronger claims are made here.
