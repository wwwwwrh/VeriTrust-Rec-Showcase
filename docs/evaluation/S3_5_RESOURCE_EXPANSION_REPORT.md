# S3.5 Resource Expansion Report

## Status

- Evaluation date: 2026-09-16
- Branch: `feature/resource-expansion-200`
- Source code commit recorded by the frozen build: `66cf8131e76999b91f25800cf27f4fc36b615fde`
- Result: final curated isolated build accepted and promoted to the canonical data paths
- Scope: bounded GitHub expansion from 20 to 200 canonical Resources

This report records the frozen data identity and acceptance evidence for the
Summer Edition S3.5 dataset. Live GitHub Search is not deterministic;
reproducibility begins at the frozen, self-contained raw run below.

## Frozen Dataset Identity

| Component | Identity |
|---|---|
| Frozen raw run ID | `a2aa04a6225e406faeb28cfcfe5d6c29` |
| Frozen raw run aggregate hash | `e7eec0a2a2498ad7c67aa716ae11df15d757ed1510df4966e27f35e36665cbed` |
| Curation manifest | `summer_s35_stage_c_final_v1` |
| Curation manifest version | `1.0.0` |
| Curation canonical hash | `sha256:0346f405c22dc339e0d3abfc043ae05df6a4373b43969bae11428c2def9ac9cb` |
| Resource Build ID | `build_60c4f8a08dad8614` |
| Evidence Build ID | `evidence_build_10deed019a096c1a` |
| Build-time Git commit | `66cf8131e76999b91f25800cf27f4fc36b615fde` |
| Intended release tag | `summer-s35-resource-200` (pending final Git publication) |

The raw run, curation manifest, Resource Build ID, Evidence Build ID, and final
Git commit/tag together form the S4 frozen dataset identity. The tag is recorded
as pending because S3.5 canonical promotion was explicitly performed without a
commit, push, merge, or tag operation.

## Collection and Selection

The `stage_c_205` profile used the frozen Stage B raw run as its baseline and
materialized a new self-contained Stage C run. Phase 2 and Phase 3 need only this
Stage C run; they do not need to dereference the baseline run.

| Metric | Result |
|---|---:|
| Unique repositories discovered | 413 |
| Selected raw repositories | 205 |
| Baseline repositories materialized | 105 |
| Newly fetched repositories | 100 |
| Metadata snapshots | 205 |
| Available README snapshots | 205 |
| README failures | 0 |
| GitHub requests | 220 |
| Cache hits / conditional revalidations | 0 / 0 |
| Retries | 0 |
| Collection wall-clock time | 114.622 s |

Final selection by query group was: core ML 14, data/scientific 17, education
22, frameworks 40, generative AI 15, MLOps 19, NLP/LLM 31, reinforcement
learning 11, vision 16, plus 20 preserved baseline records reported as
`unknown` because their older provenance predates query-group labeling.

At the end of collection the recorded GitHub rate-limit state was 4,799/5,000
remaining for the core bucket and 11/30 remaining for the search bucket. No
authentication, identity, or retry failure occurred.

## Manual Quality Control and Curation

The final manual audit contained 40 Resources:

- all 13 pre-curation weak-relevance candidates;
- 27 non-weak Resources selected with a fixed seed and stratified by query group.

Review decisions were advisory until the maintainer explicitly approved five
exact `resource_id` exclusions. All remaining `KEEP` and `REVIEW` Resources were
retained. No repository-name blacklist, fuzzy match, implicit exclusion rule, or
canonical schema field was added.

| Excluded Resource ID | Reason |
|---|---|
| `github:132750724` | General programming anthology; AI/ML content is incidental. |
| `github:21737465` | Generic cross-domain resource index rather than an AI/ML learning resource. |
| `github:266417772` | Internship listing rather than a learning resource. |
| `github:39464018` | Business-intelligence/visualization platform without a primary AI/ML learning scope. |
| `github:60493101` | General software-engineering interview curriculum rather than AI/ML learning material. |

The exact-ID curation was applied before Phase 2 canonical export. Phase 3 was
then rebuilt exclusively from the curated 200-resource dataset. All five IDs are
absent from Resources, Evidence Documents, and Evidence Chunks. The frozen raw
snapshots remain unchanged and retained.

Eight retained Resources remain in the review-only weak-relevance output:
five have README-only matches and all eight score at or below 0.30. These flags
are audit signals, not admission decisions.

## Canonical Dataset Quality

| Metric | Result |
|---|---:|
| Resources | 200 |
| Evidence Documents | 400 |
| Evidence Chunks | 5,902 |
| Metadata Evidence coverage | 100% |
| README Evidence coverage | 100% |
| Any-Evidence coverage | 100% |
| Duplicate Resource IDs | 0 |
| Orphan Documents | 0 |
| Orphan Chunks | 0 |
| Unknown Resource references | 0 |
| Provenance completeness | 100% |
| Archived Resources retained | 11 |
| Fork Resources | 0 |

The final isolated build and a second deterministic rebuild produced identical
Resource Build IDs, Evidence Build IDs, Resource ordering, Document ordering,
and Chunk ordering. Canonical promotion reproduced those same target Build IDs.

## Build and Validation Performance

Timings are wall-clock observations from this Windows development machine, not
portable performance claims.

| Operation | Time |
|---|---:|
| Phase 2 final curated build | 1.582 s |
| Phase 2 validation | 0.723 s |
| Phase 3 final curated build | 28.797 s |
| Phase 3 validation | 3.571 s |
| Expansion validation | 2.080 s |
| S1 smoke | 1.097 s |

Canonical core artifact sizes after promotion are approximately 8.91 MiB for
`data/processed`, 19.27 MiB for `data/evidence`, and 28.24 MiB when the two build
manifests are included. These figures measure different outputs from the earlier
small report-file-only metric and must not be compared as if they used the same
directory scope.

## Final S1–S3 Runtime Acceptance

Twelve representative requests were run against the canonical 200-resource
corpus with local Ollama and `qwen3:8b`: general ML, PyTorch, TensorFlow/Keras,
NLP/Transformer, LLM/RAG, computer vision, ViT, diffusion, reinforcement
learning, MLOps, beginner/project learning, and an out-of-domain cooking query.

| Runtime check | Result |
|---|---:|
| Domain queries accepted | 11/11 |
| Out-of-domain query | `RECOMMENDATION_ABSTAINED / NO_CANDIDATE` |
| Successful grounded Final Generations | 11/11 attempted |
| Same-resource Claim/Evidence binding | 100% |
| Returned chunk IDs in canonical corpus | 100% |
| Ranking mutations | 0 |
| Hallucinated Resource IDs | 0 |
| Hallucinated Claim IDs | 0 |
| Hallucinated Evidence IDs | 0 |
| Newly added S3.5 Resources appearing in Top-K | 35 unique IDs |

Representative new entries included `tpot`, `smartcore`, `pytorch-CycleGAN-and-pix2pix`,
`Mask_RCNN`, `awesome-nlp`, `LightRAG`, `vit-pytorch`, `diffusers`,
`ml-agents`, and `mlops-zoomcamp`. This is a functional smoke test, not a
relevance benchmark or a claim of optimal ranking quality.

The complete offline suite passed with `142 passed`. A real headless Streamlit
server returned HTTP 200 without an app execution exception. Existing AppTest
coverage verifies initial rendering, tabs, submit-once/rerun behavior, and clear
state behavior. The Web application displays the same canonical Evidence and
backend-grounded explanation produced by the accepted runtime.

## Known Limitations

- All 200 canonical Resources are GitHub community repositories; this is not a
  balanced multi-platform resource catalog.
- Search discovery is live and therefore not deterministic. Only the frozen raw
  run and downstream builds are reproducible.
- Weak-relevance flags and manual curation reduce obvious noise but do not prove
  comprehensive topical relevance or educational quality.
- Archived Resources are intentionally retained and exposed as soft status, not
  silently deleted.
- Phase 2 semantic fields remain conservative; README text is not used to invent
  canonical framework, skill, prerequisite, task, or difficulty facts.
- `heuristic_v0` source reliability and the Summer confidence score are
  non-calibrated heuristics, not probabilities or trustworthiness scores.
- Candidate retrieval is TF-IDF, Verification is rule-based, and local Qwen is
  restricted to Query Understanding and grounded explanation. This report does
  not evaluate Embedding retrieval, learned verification, calibrated
  uncertainty, or research-scale recommendation quality.
