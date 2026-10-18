# Documentation

Start with the [English README](../README.md) or [中文首页](../README.zh-CN.md).
Implemented Summer Edition behavior is distinct from future research.

## Understand and run

- [Architecture](ARCHITECTURE.md): data products, retrieval, claims, verification,
  heuristic confidence, decisions and the local model's role.
- [Data setup](DATA_SETUP.md): collect a small dataset, pin inputs, build both
  data products and review quality before recommending.
- [Development](DEVELOPMENT.md): environment, public interfaces, tests and
  contribution boundaries.
- [Reproducibility](REPRODUCIBILITY.md): original, standalone and fresh-clone
  validation scopes; historical benchmark availability and limits.
- [Attribution](../AUTHORS.md): shared contributions and license status.

## Evaluation and limitations

These reports preserve original methods, results, failures and limitations:

- [S2: local Query Understanding](evaluation/S2_QUERY_UNDERSTANDING_BENCHMARK.md).
- [S3: grounded explanation and UI](evaluation/S3_FINAL_GENERATION_BENCHMARK.md).
- [S3.5: Resource/Evidence expansion](evaluation/S3_5_RESOURCE_EXPANSION_REPORT.md).
- [S4-Lite: retrieval, verification and runtime](evaluation/S4_LITE_REPORT.md).

Their source commits, branches/tags and paths are historical context, not history
or data shipped here. S4-Lite used 200 Resources / 400 Documents / 5,902 Chunks;
fresh Search is not an exact reproduction. See [benchmark availability](../evaluation/benchmarks/README.md):
the quoted Verification Gold is not supplied.
