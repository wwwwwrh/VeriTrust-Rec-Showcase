# Development and contribution guide

## Environment and interfaces

Use Conda/Anaconda or Miniconda with Python 3.11 and `environment.yml`;
`requirements.txt` declares Python dependencies. Select that interpreter in VS
Code. Install Ollama/model weights separately only for real inference.

| Interface/module | Role |
|---|---|
| `data_pipeline.collectors` | GitHub REST snapshots, cache and bounded retry/rate-limit handling |
| `data_pipeline.resource_dataset` | Canonical Resources and field-level provenance |
| `data_pipeline.evidence` | Canonical documents/chunks and exact source spans |
| `recommender.run_core_pipeline` | StructuredIntent-to-core API; no LLM required |
| `recommender.pipeline.recommend_from_query` | Local query parsing followed by the core |
| `recommender.pipeline.run_user_pipeline` | Core decisions followed by grounded explanation |
| `streamlit_app.py`, `app/` | Stable root Web entrypoint and UI components |
| `evaluation/` | Historical evaluator; formal inputs are not all distributed |

See [architecture](ARCHITECTURE.md), [data setup](DATA_SETUP.md) and
[reproducibility](REPRODUCIBILITY.md). Source checkout does not install data/models.

## Run the core without an LLM

After preparing Resources and their matching Evidence Corpus:

```python
from recommender import StructuredIntent, run_core_pipeline
from recommender.config import load_config

config = load_config()
config = config.model_copy(update={
    "logging": config.logging.model_copy(update={"save_sessions": False})
})
result = run_core_pipeline(
    StructuredIntent(topics=["Computer Vision"], framework_preferences=["PyTorch"]),
    config=config,
)
print(result.model_dump_json(indent=2))
```

Start the Web UI with `python -m streamlit run streamlit_app.py`. Natural-language
CLI and Ollama setup are in the root README.

## Test changes

```powershell
python -m pip check
python -m pytest -q
git diff --check
git diff --cached --check
git status --short
```

Offline tests use temporary fixtures/mocks, not GitHub, generated data, a local
model or GPU. Mock behavior is not evidence of Qwen's semantic quality. Real-model
evaluation needs the appropriate inputs; historical metrics additionally need
the undistributed authorized archive.

On Windows, if the OS temp directory is not writable, choose a fresh disposable
directory owned by the current executing identity:

```powershell
python -m pytest -q --basetemp=.pytest-tmp/local-user -p no:cacheprovider
```

Pytest may clear it: never choose source/data. Do not share basetemp across
sandbox/desktop accounts or alter ACLs. Keep checkout/temp paths short. Pytest
retains project-root imports and importlib mode; its cache provider is disabled.

## Contribute and release safely

Use a focused branch, explain the change, run tests and inspect diffs before a
PR. Preserve [contributor attribution](../AUTHORS.md). Do not commit `.env`,
credentials, raw/generated corpora, models, logs, caches or test-temp directories.
Use relative paths/configuration instead of machine-specific settings. Review
third-party redistribution rights separately.

Preserve numeric identity, exact Evidence spans, same-resource binding, metadata
citations and backend-owned decisions. Do not tune against frozen Gold, turn
unknown into false, or present confidence/source priors as calibrated probability.
Query errors are not abstention; logging errors are warnings. UI reruns and
Top-K must not mutate shared state or silently repeat inference.

Before release, test an independent checkout without hidden data/cache/Python
paths and report actual collection/model scope. Visibility, history changes,
pushing and license selection require maintainer authorization, not just tests.
