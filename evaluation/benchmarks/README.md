# Benchmark availability

The Retrieval and Abstention YAML files contain project-authored queries, decisions and identifiers. They are retained as historical artifacts, not new measurements.

The original `s4_lite_verification.jsonl` is deliberately absent: its 15 cases quote 39 Evidence references (24,824 characters) from 14 repositories whose redistribution conditions are not fully cleared. The internal Gold-confirmation handoff is also excluded. Neither the frozen 200-resource corpus nor raw README snapshots is distributed.

The [S4-Lite report](../../docs/evaluation/S4_LITE_REPORT.md) preserves the historical results. Exact reproduction requires separately authorized benchmark content, the matching frozen corpus and original experiment Git identity. This fresh-history snapshot is not that archive. Do not manufacture its tag or substitute synthetic examples to claim the same metrics.

Public-snapshot tests use small, self-authored synthetic strings for schema, identity, binding and hash behavior. Those fixtures are not human Gold, do not establish model/verifier accuracy, and are never written into a replacement formal benchmark. A missing Verification input produces an explicit error instead of a silent fallback.
