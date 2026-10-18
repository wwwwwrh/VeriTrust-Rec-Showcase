# Reproducibility and validation scope

Historical experiments, standalone checks and actual fresh-clone acceptance
have different scopes. Equal counts do not establish identical inputs or code.
This clean-history preparation has not yet become a final committed release;
its own post-commit acceptance must be recorded separately after final README
and screenshot integration.

## Distributed inputs

The repository supplies software, configuration, project-authored fixtures and
four unchanged historical reports. It does not distribute the frozen dataset,
raw README snapshots, generated Evidence, model weights or quoted Verification
Gold. See [benchmark availability](../evaluation/benchmarks/README.md).

The omitted Verification Gold has 15 cases and 39 Evidence references totaling
24,824 characters from 14 repositories whose redistribution conditions were not
fully cleared. Synthetic fixtures do not replace that Gold or reproduce its
accuracy. Historical labels/results were not rewritten. Attribution is retained
in [AUTHORS](../AUTHORS.md); no project-wide reuse license is currently provided.

## Separate validation records

| Record | Actual scope | Boundary |
|---|---|---|
| Original Summer Edition/source baseline | 173 tests passed; the original pre-export workflow also checked 20 Resources / 40 Documents / 644 Chunks | Not the adapted standalone suite or its committed fresh clone |
| Original frozen S3.5/S4-Lite experiment | Curated 200 Resources / 400 Documents / 5,902 Chunks and human-confirmed benchmark | These data are not shipped here |
| Independent standalone snapshot, 2026-10-18 | 177 tests; a separate 20/40/644 rebuild; real local Qwen/Streamlit checks | Not proof of equivalent results on future live Search |
| Prior standalone snapshot's post-commit fresh clone, 2026-10-19 | New Conda Python 3.11.17 environment; `177 passed in 12.30s`; dependency and CLI/UI entrypoint checks | No new GitHub collection or real Qwen inference in that fresh clone; not yet acceptance of this final clean-history version |

Original methods, measurements, failures and limits remain in the unchanged
[S2](evaluation/S2_QUERY_UNDERSTANDING_BENCHMARK.md),
[S3](evaluation/S3_FINAL_GENERATION_BENCHMARK.md),
[S3.5](evaluation/S3_5_RESOURCE_EXPANSION_REPORT.md), and
[S4-Lite](evaluation/S4_LITE_REPORT.md) reports. Their original commits,
branches/tags and dataset paths describe those experiments, not inherited
history or inputs available in this repository. The S3.5 intended tag was pending
at that report's date; S4-Lite records the actual frozen tag. Do not manufacture
it here to bypass an identity check.

### Why 173 became 177

| Test category | Original | Adapted standalone suite |
|---|---:|---:|
| Unchanged test cases | 169 | 169 |
| Retained Claim-ID and benchmark-hash tests with synthetic input | 3 | 3 |
| Real Gold count/label-support check | 1 | 0 |
| Synthetic label-enum cases | 0 | 3 |
| Missing-Gold and no-substitution guards | 0 | 2 |
| Total | 173 | 177 |

The three retained tests check Claim-ID rejection, Verification input hashing and
Abstention input hashing. The real-Gold integrity assertion is no longer run in
the standalone suite. The synthetic payload's `annotation_status` only exercises
the schema, not a human annotation claim. Accepted 177-pass runs had no hidden
skips. Counts are observations, not a number to force by altering expectations.

## Independent small-data rebuild

The 2026-10-18 standalone run began without copied data, API cache or `.env`.
The `baseline_20` collector used private process-environment authentication:
47 requests, 0 retries/cache hits, 20 metadata snapshots and 20 README snapshots.
Both builders consumed the explicitly pinned self-contained run.

| Item | Recorded value |
|---|---|
| Raw run ID | `fb8e67e85a344968b4c959a516b8c710` |
| Resource Build | `build_ab7b0d3c7569525d` |
| Evidence Build | `evidence_build_8164348f3258a9a5` |
| Resources / Documents / Chunks | 20 / 40 / 644 |
| Metadata / README / any-Evidence coverage | 100% / 100% / 100% |
| Invalid schema / failed parse / duplicate merges | 0 / 0 / 0 |
| Orphan Documents / Chunks | 0 / 0 |
| Evidence provenance completeness | 100% |

These generated records remain local-only. Human review retained broad matches
and a legitimate handbook flagged by the limited relevance vocabulary; no
admission rule was tuned to beautify results. Default config contains historical
scale/reference-time labels: report actual counts, inputs and build identities,
not a dataset size inferred from a label.

### Real Qwen and Web checks in that workflow

Using local Ollama 0.34.0 / `qwen3:8b`:

- `I want to learn Computer Vision with PyTorch.` returned `OK`, Final Generation
  `SUCCESS`, valid grounding and 3 recommendations. All 28 returned Evidence
  references resolved to exact canonical text and the same Resource as the Claim.
- `I want to learn cooking and baking bread.` returned
  `RECOMMENDATION_ABSTAINED / NO_CANDIDATE`, with Final Generation `NOT_RUN`.
- Backend ranking, fact text and Claim/Evidence identities were preserved.
- Streamlit HTTP root/health returned 200. A real-Qwen AppTest exercised Evidence,
  four tabs, result reuse on rerun, Clear and abstention without execution errors.
  AppTest is not a browser screenshot.

These are functional observations, not general recommendation-quality or
hallucination-free guarantees. They are distinct from the later fresh clone,
which checked initial rendering without submitting a real-model query.

## Reproduce the software workflow

In a new checkout:

```powershell
conda env create -f environment.yml
conda activate veritrust-rec
python -m pip check
python -m pytest -q
```

Ordinary tests use mocks and temporary synthetic fixtures, including offline
builder integration tests. They need no GPU, Ollama, GitHub access or existing
generated dataset. Select the actual interpreter in VS Code; do not rely on
`PYTHONPATH` pointing to another checkout. On Windows use a fresh disposable
test-temp directory owned by the executing identity if needed; do not share
it between sandbox/desktop accounts or change system ACLs.

For real recommendations follow [data setup](DATA_SETUP.md): collect
`baseline_20`, pin the run, build/validate Resources, review quality, then
build/validate matching Evidence. Direct S1 needs data but no LLM; raw-query
parsing/final explanation additionally need local Ollama. See
[development](DEVELOPMENT.md) and [architecture](ARCHITECTURE.md).

## Reproduction limits and final acceptance

Live GitHub Search is not deterministic. Reproducibility starts at frozen raw
inputs, effective configuration and versions. New runs need their own run/build
IDs, dependency/model versions, evaluation time and hardware record. New data
need not produce 644 chunks.

Exact S4-Lite reproduction requires authorized original Gold, the matching
frozen corpus and original experiment identity. Its evaluator rejects arbitrary
rebuilt datasets. Neither fabricated tags nor synthetic replacements establish
the historical metrics. Dependency ranges and local runtime can also change;
record actual versions and outcomes.

For final clean-history acceptance, independently clone its root commit, install
dependencies, test, inspect entrypoints/links/images and audit Git objects. Verify
that no old dataset or Python path supplied hidden inputs. Report separately
whether real collection/model inference were repeated; a mock-only pass is not
a Qwen benchmark. Tests and source links do not clear third-party distribution
rights or authorize changing repository visibility.
