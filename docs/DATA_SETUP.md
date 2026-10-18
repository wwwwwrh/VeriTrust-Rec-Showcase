# Data setup and reproducibility

For validation records and the limits of reproduction from this source-only
snapshot, see [reproducibility and validation scope](REPRODUCIBILITY.md).

This guide is the supported starting point for a new clone. Run commands from the repository root in the Python 3.11 Conda environment. Generated data, raw snapshots, API caches and logs are local-only and are ignored by Git.

**Two separate workflows:** build your own small dataset from GitHub (A), or explicitly restore an already-held accepted frozen copy (B). A fresh clone does not contain the inputs for B. No command silently falls back to another dataset.

## A. Build a small local dataset

### 1. Environment and optional token

```powershell
conda env create -f environment.yml
conda activate veritrust-rec
Copy-Item .env.example .env
```

Edit `.env` locally if using a token. Do not print it, paste it into a command argument, commit it, or send it to a collaborator. The collector reads `GITHUB_TOKEN` from the environment or local `.env`. `baseline_20` can use unauthenticated public reads but is subject to much lower GitHub rate limits.

A token does not grant permission to redistribute README contents. Review each source's applicable licenses and attribution requirements separately.

### 2. Collect and pin one raw run

```powershell
$collectorOutput = python -m data_pipeline.collectors.github_collector --profile baseline_20
if ($LASTEXITCODE -ne 0) { throw "Collector failed. Inspect its log before proceeding." }
$collectorOutput
$summary = ($collectorOutput -join [Environment]::NewLine) | ConvertFrom-Json
$run = Join-Path "data/raw/github" ($summary.retrieved_at.Substring(0, 10) + "/" + $summary.run_id)
$run
Get-Content (Join-Path $run "validation_summary.json")
```

Use this exact `$run` for both builders. Do not choose the newest directory automatically. Store the run ID in your local notes if opening another terminal.

`configs/github.yaml` controls profiles, limits, query groups, timeout, bounded retry/backoff, rate-limit handling and caching. The default profile targets **at most 20** repositories; it is not the 200-resource frozen experiment. Search is live and can change. Numeric GitHub repository IDs are the stable identity; names/owners can change.

The collector appends:

```text
data/raw/github/<UTC-date>/<run-id>/
  github_search/
  github_repository/
  github_readme/
  validation_summary.json
cache/github/
logs/
data/failed_fetches.jsonl
```

Before continuing, confirm the summary has a nonzero `repositories` count, no authentication or identity failure, and the metadata files match the reported IDs/count. Read failure logs if collection stopped or returned fewer than the target. Do not treat a directory's existence as a successful run. README fetch failures are isolated and must be reported, not repaired with synthetic text.

### 3. Phase 2 build and validation

These commands write your local canonical paths. On an existing installation, use a separate clone/workspace if you want to preserve its current corpus.

```powershell
python -m data_pipeline.resource_dataset.build --input-run $run
if ($LASTEXITCODE -ne 0) { throw "Phase 2 build failed." }
python -m data_pipeline.resource_dataset.validate
if ($LASTEXITCODE -ne 0) { throw "Phase 2 validation failed." }
python -m data_pipeline.resource_dataset.report
if ($LASTEXITCODE -ne 0) { throw "Phase 2 report failed." }
```

Read these outputs **before proceeding**:

```text
data/metadata/build_manifest.json
data/reports/resource_dataset_quality.json
data/reports/resource_dataset_report.md
data/quarantine/invalid_schema.jsonl
data/quarantine/failed_parse.jsonl
data/review/low_relevance.jsonl
data/review/duplicate_candidates.jsonl
```

Check Resource count against valid raw metadata, unique `github:<numeric-id>` IDs, timestamps/URLs, provenance and all quarantine reasons. Manually inspect the small dataset's descriptions/topics/source links, especially low-relevance or duplicate-review candidates. Low relevance, archived/fork status and unknown difficulty remain **review-only / soft flags**, not automatic exclusion. Even unflagged records may be off-domain; automatic validation is not a substitute for human review.

Do not hand-edit JSONL/Parquet or delete raw snapshots. If the dataset is unusable, stop and investigate the inputs/configuration. The historical five-ID curation manifest is specific to the frozen Stage C run and is **not** an admission policy for new runs.

Outputs include `resources.jsonl`, `resources.parquet`, aliases and field-level provenance under `data/processed/`. They have stable ordering and deterministic build identities for the same frozen inputs, effective config and versions. The default config retains historical reference-time/scale labels; report your actual record count and config rather than interpreting that label as a 200-resource claim.

### 4. Phase 3 Evidence build and validation

```powershell
python -m data_pipeline.evidence.build --resources data/processed/resources.parquet --input-run $run --config configs/evidence_corpus.yaml
if ($LASTEXITCODE -ne 0) { throw "Phase 3 build failed." }
python -m data_pipeline.evidence.validate --resources data/processed/resources.parquet
if ($LASTEXITCODE -ne 0) { throw "Phase 3 validation failed." }
python -m data_pipeline.evidence.report
if ($LASTEXITCODE -ne 0) { throw "Phase 3 report failed." }
```

Inspect:

```text
data/evidence/documents.jsonl
data/evidence/chunks.jsonl
data/metadata/evidence_build_manifest.json
data/reports/evidence_corpus_quality.json
data/reports/evidence_corpus_report.md
data/quarantine/evidence/
data/review/evidence/
```

Require valid schema, matching Resource/Evidence builds, no orphan/unknown references, exact source offsets and complete provenance. Metadata/any-Evidence coverage must match admitted resources. Investigate missing README Evidence; do not synthesize it or replace a good resource merely to meet a coverage target.

README and repository metadata become separate Evidence documents. Identical text at different source positions is preserved. Oversized intact code/table chunks may require review; ordinary prose must meet the validator's constraints. Build validation failures must be investigated, not bypassed by changing chunking or suppressing checks.

Both builders stage and validate outputs before publishing, with manifests as commit markers. This is not a cross-Phase transaction: do not run the recommender between a new Phase 2 build and the matching Phase 3 build. Preserve a separate known-good workspace when experimenting.

### 5. Run

Use the CLI and Streamlit commands in the [README](../README.md#quick-start), or the [direct S1 API example](DEVELOPMENT.md#run-the-core-without-an-llm). CLI natural-language parsing requires local Ollama; the S1 API does not.

The six runtime inputs are:

```text
data/processed/resources.jsonl
data/processed/resource_provenance.jsonl
data/metadata/build_manifest.json
data/evidence/documents.jsonl
data/evidence/chunks.jsonl
data/metadata/evidence_build_manifest.json
```

A Web page loading does not prove those data exist: resources are loaded when needed. If a request reports a missing Resource Dataset or Evidence Corpus, finish this guide first.

## B. Restore an existing accepted frozen copy (optional)

For maintainers who already possess the accepted copy, the explicit command is:

```powershell
python -m data_pipeline.runtime.materialize --source-root data/s35/stages/stage-c-final/data
```

That path is only an example of a **locally present source**, not a downloadable artifact or a clone prerequisite. Specify your actual source root. The materializer requires Resource Build `build_60c4f8a08dad8614`, Evidence Build `evidence_build_10deed019a096c1a`, the accepted curation hash and counts 200/400/5,902. It validates source identity and refuses conflicting existing targets; it does not recollect, rebuild or fall back to a different directory. It copies only six runtime files, not the Phase 2 Parquet or complete build/report tree.

Do not use it to install a newly collected dataset or overwrite mismatching generated files. In particular, equivalent build IDs do not authorize overwriting non-identical file bytes. Resolve conflicts explicitly in a separate workspace.

## Reproduction boundary and content rights

Reproducibility starts at a frozen raw run, not GitHub Search. Record raw run ID, effective configuration, Resource/Evidence build IDs, curation hash if any, code commit and dirty/code provenance. Different search results cannot reproduce the historical S4-Lite benchmark by name alone.

Formal S4-Lite reproduction requires the separately authorized **exact frozen corpus**, human-confirmed Verification benchmark and original experiment archive/Git identity. The corpus and quoted Verification Gold are not distributed with this independent snapshot pending rights clearance; see [benchmark availability](../evaluation/benchmarks/README.md). The original evaluator preflight deliberately rejects arbitrary rebuilt datasets and does not manufacture the absent historical tag. See the [historical report](evaluation/S4_LITE_REPORT.md); do not relabel new live-run metrics or synthetic engineering fixtures as frozen results.

No full corpus download is supplied. README text, images and source metadata can have distinct rights/terms. Keeping generated files out of Git does not clear third-party text already in versioned benchmarks or history. Redistribution requires a separate review of applicable scope, permissions and required notices; successful data validation is not license clearance.
